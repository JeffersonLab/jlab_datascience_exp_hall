import tensorflow as tf
from jlab_datascience_toolkit.core.jdst_model import JDSTModel
import aidapt_toolkit.utils.gradient_monitor as gm
from tensorflow.keras.callbacks import Callback
from omegaconf import OmegaConf
from aidapt_toolkit.registration import make
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
import yaml
import os
import copy


# Use to force running in eager mode
# tf.config.run_functions_eagerly(True)

def get_optimizer(type: str = "Adam", optimizer_config: dict = None):
    """Gets a non-legacy Optimizer instance

    Args:
        optimizer_config (dict, optional): Optional configuration dictionary.
            When not provided, returns an Adam optimizer with default settings.
    Returns:
        tf.keras.optimizer.Optimizer
    """

    if optimizer_config is None:
        optimizer_config = dict()

    # Check if a learning rate schedule is defined
    if "learning_rate_schedule" in optimizer_config:
        lr_schedule_config = optimizer_config.pop("learning_rate_schedule")
        print("lr_schedule_config: ", lr_schedule_config)
        lr_type = lr_schedule_config.pop("type")
        if lr_type == "ExponentialDecay":
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr_schedule_config["initial_learning_rate"],
                decay_steps=lr_schedule_config["decay_steps"],
                decay_rate=lr_schedule_config["decay_rate"],
                staircase=lr_schedule_config.get(
                    "staircase", False
                ),  # Default to False if not provided
            )
            print(
                "lr_schedule_config['decay_steps']: ", lr_schedule_config["decay_steps"]
            )
        if lr_type == "PolynomialDecay":
            learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=lr_schedule_config["initial_learning_rate"],
                decay_steps=lr_schedule_config["decay_steps"],
                end_learning_rate=lr_schedule_config["end_learning_rate"],
                power=lr_schedule_config.get(
                    "power", 1.0
                ),  # Default to False if not provided
            )
        else:
            raise ValueError(f"Unsupported learning rate schedule type: {lr_type}")

        # Set the learning rate schedule in the optimizer configuration
        optimizer_config["learning_rate"] = learning_rate

    module = "keras.optimizers"
    return tf.keras.optimizers.deserialize(
        dict(class_name=type, module=module, config=optimizer_config)
    )

def get_layer(type: str = "Dense", layer_config: dict = None):
    """Gets a non-legacy Layer instance

    Args:
        layer_config (dict): Optional configuration dictionary.
            When not provided, returns a Dense layer with a single unit output and default settings.

    Returns:
        tf.keras.layers.Layer
    """
    if layer_config is None:
        layer_config = dict(units=1)

    module = "keras.layers"
    return tf.keras.layers.deserialize(
        dict(class_name=type, module=module, config=layer_config)
    )


def build_sequential_model(layers: list = None):
    if layers is None:
        raise RuntimeError("Cannot build model without a list of layers")

    layer_list = []
    for layer_config in layers:
        # layer_config is a tuple of (type, config_dict)
        type, config_dict = layer_config
        layer_list.append(get_layer(type=type, layer_config=config_dict))

    return tf.keras.Sequential(layer_list)


class BatchHistory(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}
        self.current_batch = tf.Variable(0, dtype=tf.int32)
        self.current_epoch = tf.Variable(0, dtype=tf.int32)

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)
        self.model.batch_history = self

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch.assign(epoch)

    def on_batch_begin(self, batch, logs=None):
        self.current_batch.assign(batch)


class TF_CGAN_Keras(tf.keras.Model):
    def __init__(self, discriminator, generator, noise_dim=100, batch_size=32):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

        self.discriminator_gradient_norm_tracker = tf.keras.metrics.Mean(
            name="discriminator_gradient_norm"
        )
        self.generator_gradient_norm_tracker = tf.keras.metrics.Mean(
            name="generator_gradient_norm"
        )
        self.gradient_monitor = None
        self.generator_gradients = None
        self.discriminator_gradients = None
        self.steps_per_epoch = None
        self.k = None
        self.tracker = None
        self.initialized = False

        # Initialize discriminator accuracy trackers
        self.disc_acc_tracker = tf.keras.metrics.Mean(name="disc_acc")
        self.true_labels_list = []
        self.predicted_scores_list = []

        # Initialize discriminator gradient norm trackers for each layer
        self.discriminator_layer_gradient_norms_tracker = {}
        disc_layer_counter = 0
        for layer in discriminator.layers:
            if layer.trainable_weights:
                if isinstance(layer, tf.keras.Sequential):
                    disc_layer_counter += 1
                    for sub_layer in layer.trainable_weights:
                        self.discriminator_layer_gradient_norms_tracker[
                            f"disc_{sub_layer.path}"
                        ] = tf.keras.metrics.Mean(
                            name=f"disc_{sub_layer.path}_grad_norm"
                        )
                else:
                    for sub_layer in layer.trainable_weights:
                        self.discriminator_layer_gradient_norms_tracker[
                            f"disc_{sub_layer.path}"
                        ] = tf.keras.metrics.Mean(
                            name=f"disc_{sub_layer.path}_grad_norm"
                        )

        # Initialize generator gradient norm trackers for each layer
        self.generator_layer_gradient_norms_tracker = {}
        for layer in generator.layers:
            if layer.trainable_weights:
                if isinstance(layer, tf.keras.Sequential):
                    for sub_layer in layer.trainable_weights:
                        self.generator_layer_gradient_norms_tracker[
                            f"gen_{sub_layer.path}"
                        ] = tf.keras.metrics.Mean(
                            name=f"gen_{sub_layer.path}_grad_norm"
                        )

                else:
                    for sub_layer in layer.trainable_weights:
                        self.generator_layer_gradient_norms_tracker[
                            f"gen_{sub_layer.path}"
                        ] = tf.keras.metrics.Mean(
                            name=f"gen_{sub_layer.path}_grad_norm"
                        )

    def set_tracker(self, tracker):
        self.tracker = tracker

    def compile(self, d_optimizer, d_loss_fn, g_optimizer, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    @tf.function
    def pretrain_step(self, vertex_batch, ps_batch, loss_fn):
        if len(vertex_batch.shape) == 1:
            vertex_batch = tf.reshape(vertex_batch, [-1, 1])
        batch_size = tf.shape(vertex_batch)[0]
        random_noise = tf.random.normal(shape=(batch_size, self.noise_dim))
        with tf.GradientTape() as tape:
            generated_data = self.generator([vertex_batch, random_noise], training=True)
            loss = loss_fn(ps_batch, generated_data)
        grads = tape.gradient(loss, self.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return loss

    def pretrain_outer_generator(
        self, realistic_vertex_data, ps_vertex_data, epochs=10, batch_size=2048
    ):
        """
        Pre-trains `outer_generator` to produce outputs resembling the phasespace data
        on which the inner GAN was trained.
        """
        print("Running pre-training of unfolding generator")
        vertex_s = ps_vertex_data[:, 4]
        ps_vertex_data = ps_vertex_data[:, :4]
        dataset = tf.data.Dataset.from_tensor_slices((vertex_s, ps_vertex_data))
        dataset = dataset.batch(batch_size)
        pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
        mse_loss_fn = tf.keras.losses.MeanSquaredError()
        for epoch in range(epochs):
            for step, (vertex_batch, ps_batch) in enumerate(dataset):
                loss = self.pretrain_step(vertex_batch, ps_batch, mse_loss_fn)
            # print(f"Epoch {epoch + 1}/{epochs} of unfolding generator pre-training, Loss: {loss.numpy():.4f}")

    def train_step(self, data):
        inputs, real_images = data
        batch_size = tf.shape(inputs)[0]

        noise = tf.random.normal(shape=(batch_size, self.noise_dim))  # , stddev=10)
        generated_images = self.generator((inputs, noise))

        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_inputs = tf.concat([inputs, inputs], axis=0)

        # Label fake images and real images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        # Store the true labels for accuracy calculation before adding noise
        true_labels = labels

        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        disc_updates = 5
        # Train discriminator
        for _ in range(disc_updates):
            with tf.GradientTape(persistent=True) as disc_tape:
                predictions = self.discriminator((combined_inputs, combined_images))
                d_loss = self.d_loss_fn(labels, predictions)
            discriminator_gradients = disc_tape.gradient(
                d_loss, self.discriminator.trainable_weights
            )
            self.d_optimizer.apply_gradients(
                zip(discriminator_gradients, self.discriminator.trainable_weights)
            )

        # If using logits, apply sigmoid to get probabilities
        if isinstance(
            self.d_loss_fn, tf.keras.losses.BinaryCrossentropy
        ) and not getattr(self.d_loss_fn, "from_logits", False):
            pass  # No action needed
        else:
            # Apply sigmoid if predictions are logits
            predictions = tf.sigmoid(predictions)

        predicted_scores = predictions
        predicted_labels = tf.cast(predictions > 0.5, tf.float32)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted_labels, tf.round(true_labels)), tf.float32)
        )

        disc_layer_grad_norms = []
        for idx, grad in enumerate(discriminator_gradients):
            norm = tf.norm(grad)
            disc_layer_grad_norms.append(norm)

        # Calculate discriminator gradient norms
        discriminator_agg_gradient_norm = tf.sqrt(
            sum([tf.reduce_sum(tf.square(g)) for g in discriminator_gradients])
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.noise_dim)
        )  # , stddev=10)

        # Assemble labels that say "all real images"
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as gen_tape:
            predictions = self.discriminator(
                (inputs, self.generator((inputs, random_latent_vectors)))
            )
            g_loss = self.g_loss_fn(misleading_labels, predictions)
        generator_gradients = gen_tape.gradient(
            g_loss, self.generator.trainable_weights
        )
        self.g_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_weights)
        )

        gen_layer_grad_norms = []
        for idx, grad in enumerate(generator_gradients):
            norm = tf.norm(grad)
            gen_layer_grad_norms.append(norm)

        # Calculate generator gradient norms
        generator_agg_gradient_norm = tf.sqrt(
            sum([tf.reduce_sum(tf.square(g)) for g in generator_gradients])
        )

        # ----------------------------- Update tracked metrics ------------------------------- #
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.discriminator_gradient_norm_tracker.update_state(
            discriminator_agg_gradient_norm
        )
        self.generator_gradient_norm_tracker.update_state(generator_agg_gradient_norm)

        # Layer-specfic gradient norms for discriminator
        disc_item_counter = 0
        for item in self.discriminator_layer_gradient_norms_tracker:
            self.discriminator_layer_gradient_norms_tracker[item].update_state(
                disc_layer_grad_norms[disc_item_counter]
            )
            disc_item_counter += 1
            # print("item: ", item)

        # Layer-specific gradient norms for generator
        gen_item_counter = 0
        for item in self.generator_layer_gradient_norms_tracker:
            self.generator_layer_gradient_norms_tracker[item].update_state(
                gen_layer_grad_norms[gen_item_counter]
            )
            gen_item_counter += 1

        self.disc_acc_tracker.update_state(accuracy)
        self.true_labels_list.append(true_labels)
        self.predicted_scores_list.append(predictions)
        # ------------------------------------------------------------------------------------ #

        # Return tracked metrics
        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "discriminator_gradient_norm": self.discriminator_gradient_norm_tracker.result(),
            "generator_gradient_norm": self.generator_gradient_norm_tracker.result(),
        }

        # If you want to track average losses over an epoch, instead of per batch,
        # use this return call.
        # return {
        #     "d_loss": self.d_loss_tracker.result(),
        #     "g_loss": self.g_loss_tracker.result(),
        # }


class TF_CGAN(JDSTModel):
    def __init__(self, config: dict = None, name: str = "unfolding_model_v1") -> None:
        """_summary_

        Args:
            config (dict, optional): Dictionary of configuration options.
                Defaults to:
                    latent_dim=100,
                    generator_optimizer=None,
                    discriminator_optimizer=None,
                    label_shape=5,
                    image_shape=4,
                    generator_layers=None,
                    discriminator_layers=None,
                    epochs=1,
                    batch_size=32


            name (str, optional): Name of Module. Defaults to 'unfolding_model_v1'.
        """
        self.module_name = name

        # Set config defaults
        self.config = dict(
            latent_dim=100,
            generator_optimizer=None,
            discriminator_optimizer=None,
            label_shape=5,
            image_shape=4,
            inner_generator_layers=None,
            generator_layers=None,
            discriminator_layers=None,
            epochs=1,
            batch_size=32,
            random_seed=42,
            make_layer_grad_plots=False,
            make_chi2_plots=False,
            chi2_frequency=100,
            gan_type="inner",
        )

        if config is not None:
            self.config.update(config)

        self.set_model_variables()

        self.rng = tf.random.Generator.from_seed(self.config["random_seed"])

        # TODO: Make sure random_seed also controls model weights/optimizers/etc

        self.discriminator_optimizer = get_optimizer(
            *self.config["discriminator_optimizer"]
        )

        self.generator_optimizer = get_optimizer(*self.config["generator_optimizer"])

        self.gen_loss_fn = self.get_loss_function(self.config["generator_loss"])
        self.disc_loss_fn = self.get_loss_function(self.config["discriminator_loss"])

        self.acc_fn = tf.keras.metrics.binary_accuracy

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        if self.config["gan_type"].lower() == "inner":
            self.cgan = TF_CGAN_Keras(
                self.discriminator,
                self.generator,
                noise_dim=self.config["latent_dim"],
                batch_size=self.config["batch_size"],
            )
            self.cgan_model = TF_CGAN_Keras(self.discriminator, self.generator)
        elif self.config["gan_type"].lower() == "outer":

            self.cgan_model = TF_CGAN_Keras(self.discriminator, self.generator)

            assert "folding_path" in self.config
            assert "folding_id" in self.config

            # self.unfolding_model.static_generator = self.build_static_generator()

            # Load the trained folding GAN (inner)
            self.folding_path = self.config["folding_path"]
            self.folding_id = self.config["folding_id"]

            with open(f"{self.folding_path}/config.yaml", "r") as f:
                self.folding_config = yaml.safe_load(f)

            self.folding_model = make(self.folding_id, config=self.folding_config)
            self.folding_model.load_inner_GAN(self.folding_path)

            # This assumes InnerGAN is a TF_CGAN...
            self.folding_model.cgan.trainable = False

            from aidapt_toolkit.models.tf_outer_cgan_v0 import TF_OuterGAN_V0

            self.cgan = TF_OuterGAN_V0(
                self.discriminator,
                self.generator,
                self.folding_model,
                noise_dim=self.config["latent_dim"],
                batch_size=self.config["batch_size"],
            )

            # self.folding_model = TF_CGAN_Keras(self.discriminator, self.generator)
        else:
            raise ValueError("gan_type must be 'inner' or 'outer'")

        self.cgan.compile(
            self.discriminator_optimizer,
            self.disc_loss_fn,
            self.generator_optimizer,
            self.gen_loss_fn,
        )

    def get_loss_function(self, loss_name):
        # Get the loss function from tf.keras.losses by name
        try:
            return getattr(tf.keras.losses, loss_name)()
        except AttributeError:
            raise ValueError(
                f"Loss function '{loss_name}' is not a valid tf.keras.losses function"
            )

    def set_model_variables(self):
        self.latent_dim = self.config[f"latent_dim"]
        self.generator_layers = self.config[f"generator_layers"]
        self.discriminator_layers = self.config[f"discriminator_layers"]
        self.batch_size = self.config[f"batch_size"]
        self.epochs = self.config[f"epochs"]
        self.image_shape = self.config[f"image_shape"]
        self.label_shape = self.config[f"label_shape"]
        self.disc_loss_fn = self.config[f"discriminator_loss"]
        self.gen_loss_fn = self.config[f"generator_loss"]

    def build_static_generator(self):
        # label_shape = 4
        label_shape = self.config["label_shape"]
        image_shape = self.config["image_shape"]
        label = tf.keras.layers.Input(shape=(label_shape,))
        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Concatenate()([label, noise])

        model = build_sequential_model(self.config["generator_layers"])
        x = model(x)

        output = tf.keras.layers.Dense(image_shape, activation="tanh")(x)

        generator = tf.keras.models.Model(inputs=[label, noise], outputs=[output])
        return generator

    def build_generator(self):
        label_shape = self.config["label_shape"]
        image_shape = self.config["image_shape"]

        label = tf.keras.layers.Input(shape=(label_shape,))
        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Concatenate()([label, noise])

        model = build_sequential_model(self.config["generator_layers"])

        x = model(x)

        output = tf.keras.layers.Dense(image_shape, activation="tanh")(x)

        generator = tf.keras.models.Model(inputs=[label, noise], outputs=[output])

        return generator

    def build_discriminator(self):
        label_shape = self.config["label_shape"]
        image_shape = self.config["image_shape"]
        label = tf.keras.layers.Input(shape=(label_shape,))
        image = tf.keras.layers.Input(shape=(image_shape,))
 
        combined_input_shape = label_shape + image_shape
        combined_input = tf.keras.layers.Input(shape=(image_shape,))

        x = tf.keras.layers.Concatenate()([label, image])

        model = build_sequential_model(self.config["discriminator_layers"])
        x = model(x)

        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        discriminator = tf.keras.models.Model(inputs=[label, image], outputs=output)

        return discriminator

    def build_static_discriminator(self):
        label_shape = self.config["label_shape"]
        image_shape = self.config["image_shape"]

        label = tf.keras.layers.Input(shape=(label_shape,))
        image = tf.keras.layers.Input(shape=(image_shape,))

        x = tf.keras.layers.Concatenate()([label, image])

        model = build_sequential_model(self.config["discriminator_layers"])
        x = model(x)

        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        discriminator = tf.keras.models.Model(inputs=[label, image], outputs=output)

        return discriminator

    def get_info(self):
        print(inspect.getdoc(self))

    def load_config(self):
        return super().load_config()

    def save_config(self):
        return super().save_config()

    def load(self, filepath):
        # Set all configuration settings
        with open(os.path.join(filepath, "config.yaml"), "r") as f:
            self.config.update(yaml.safe_load(f))
        self.set_model_variables()

        # Build correct network before loading weights
        self.generator = self.build_generator()
        # self.static_generator = self.build_static_generator()
        self.discriminator = self.build_discriminator()

        # Load network weights
        self.generator.load_weights(os.path.join(filepath, "generator.weights.h5"))
        # self.static_generator.load_weights(os.path.join(filepath, 'generator.weights.h5'))
        self.discriminator.load_weights(
            os.path.join(filepath, "discriminator.weights.h5")
        )

        self.cgan = TF_CGAN_Keras(
            self.discriminator,
            self.generator,
            noise_dim=self.config["latent_dim"],
            batch_size=self.config["batch_size"],
        )

        self.cgan.compile(
            self.discriminator_optimizer,
            self.disc_loss_fn,
            self.generator_optimizer,
            self.gen_loss_fn,
        )
        # TODO load optimizers specified in config...

    def load_inner_GAN(self, filepath):
        # Set all configuration settings
        with open(os.path.join(filepath, "config.yaml"), "r") as f:
            self.config.update(yaml.safe_load(f))
        self.set_model_variables()

        # Build correct network before loading weights
        self.static_generator = self.build_generator()
        # self.static_generator = self.build_static_generator()
        self.static_discriminator = self.build_discriminator()

        # Load network weights
        # self.generator.load_weights(os.path.join(filepath, 'generator.weights.h5'))
        self.static_generator.load_weights(
            os.path.join(filepath, "generator.weights.h5")
        )
        # self.static_discriminator.load_weights(os.path.join(filepath, 'discriminator.weights.h5'))

        self.static_generator.trainable = False
        self.static_discriminator.trainable = False

        self.cgan = TF_CGAN_Keras(
            self.static_discriminator,
            self.static_generator,

            noise_dim=self.config["latent_dim"],
            batch_size=self.config["batch_size"],
        )

        self.cgan.compile(
            self.discriminator_optimizer,
            self.disc_loss_fn,
            self.generator_optimizer,
            self.gen_loss_fn,
        )
        # TODO load optimizers specified in config...

    def save(self, filepath):
        os.makedirs(filepath)
        self.generator.save_weights(os.path.join(filepath, "generator.weights.h5"))
        self.discriminator.save_weights(
            os.path.join(filepath, "discriminator.weights.h5")
        )

        # Create a deep copy of the config to modify
        serializable_config = copy.deepcopy(self.config)

        # Convert the ExponentialDecay schedule for discriminator_optimizer
        if "discriminator_optimizer" in serializable_config:
            optimizer_config = serializable_config["discriminator_optimizer"][1]
            if "learning_rate" in optimizer_config:
                lr_schedule = optimizer_config["learning_rate"]
                if isinstance(
                    lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay
                ):
                    optimizer_config["learning_rate"] = {
                        "type": "ExponentialDecay",
                        "initial_learning_rate": lr_schedule.initial_learning_rate,
                        "decay_steps": lr_schedule.decay_steps,
                        "decay_rate": lr_schedule.decay_rate,
                        "staircase": lr_schedule.staircase,
                    }
                elif isinstance(
                    lr_schedule, tf.keras.optimizers.schedules.PolynomialDecay
                ):
                    optimizer_config["learning_rate"] = {
                        "type": "PolynomialDecay",
                        "initial_learning_rate": lr_schedule.initial_learning_rate,
                        "decay_steps": lr_schedule.decay_steps,
                        "end_learning_rate": lr_schedule.end_learning_rate,
                        "power": lr_schedule.power,
                    }

        # Convert the ExponentialDecay schedule for generator_optimizer
        if "generator_optimizer" in serializable_config:
            optimizer_config = serializable_config["generator_optimizer"][1]
            if "learning_rate" in optimizer_config:
                lr_schedule = optimizer_config["learning_rate"]
                if isinstance(
                    lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay
                ):
                    optimizer_config["learning_rate"] = {
                        "type": "ExponentialDecay",
                        "initial_learning_rate": lr_schedule.initial_learning_rate,
                        "decay_steps": lr_schedule.decay_steps,
                        "decay_rate": lr_schedule.decay_rate,
                        "staircase": lr_schedule.staircase,
                    }
                elif isinstance(
                    lr_schedule, tf.keras.optimizers.schedules.PolynomialDecay
                ):
                    optimizer_config["learning_rate"] = {
                        "type": "PolynomialDecay",
                        "initial_learning_rate": lr_schedule.initial_learning_rate,
                        "decay_steps": lr_schedule.decay_steps,
                        "end_learning_rate": lr_schedule.end_learning_rate,
                        "power": lr_schedule.power,
                    }

        with open(os.path.join(filepath, "config.yaml"), "w") as f:
            # yaml.safe_dump(self.config, f)
            # yaml.safe_dump(serializable_config, f)
            OmegaConf.save(self.config, f)
        # TODO: Save state of optimizers

    def analysis(self, path=None):
        # Plot training history
        hist = self.cgan.batch_history.history

        plt.semilogy(np.array(hist["g_loss"]) * 4, label="Generator")
        plt.semilogy(np.array(hist["d_loss"]) * 4, label="Discriminator")
        # plt.semilogy(self.history['d_loss_fake']*4, label='Discriminator Fake')
        plt.axhline(y=1, linestyle=":", color="k")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot gradient norms
        discriminator_gradient_norms = np.array(hist["discriminator_gradient_norm"])
        generator_gradient_norms = np.array(hist["generator_gradient_norm"])
        plt.subplot(1, 2, 2)
        plt.plot(np.array(hist["discriminator_gradient_norm"]), label="Discriminator")
        plt.plot(np.array(hist["generator_gradient_norm"]), label="Generator")
        plt.ylabel("Gradient Norm")
        plt.xlabel("Epoch")
        plt.legend()

        if path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(path, "model_history.png"))

    def train(
        self,
        input_data,
        output_path,
        batches_per_epoch,
        target_distributions,
        d_scaler,
        latent_dim,
        make_layer_grad_plots,
        grad_frequency,
        make_chi2_plots,
        chi2_frequency,
        make_disc_accuracy_plots,
        accuracy_frequency,
    ):

        if self.config["gan_type"].lower() == "inner":
            detector_level_data, vertex_level_data_combined = input_data
        elif self.config["gan_type"].lower() == "outer":
            detector_level_data, (vertex_level_data, vertex_s) = input_data
            vertex_level_data_combined = tf.concat(
                [vertex_level_data, vertex_s], axis=1
            )

        tracker = BatchHistory()
        self.cgan.set_tracker(tracker)

        # Initialize gradient monitor
        gradient_monitor = gm.GradientMonitor(
            self.cgan, output_path, grad_frequency, make_layer_grad_plots
        )
        callbacks_list = [tracker, gradient_monitor]

        # Add "chi_square_monitor" to "callbacks_list" if set as "True" in config file
        if make_chi2_plots:
            if self.config["gan_type"].lower() == "inner":
                data_for_chi2 = vertex_level_data_combined
                models_for_chi2 = self.cgan
            elif self.config["gan_type"].lower() == "outer":
                data_for_chi2 = vertex_s
                models_for_chi2 = (self.cgan, self.folding_model.cgan)

            chi_square_monitor = gm.ChiSquareMonitor(
                models_for_chi2,
                data_for_chi2,
                target_distributions,
                output_path,
                latent_dim,
                d_scaler,
                output_path,
                frequency=chi2_frequency,
                gan_type=self.config["gan_type"].lower(),
            )

            callbacks_list.append(chi_square_monitor)

        # Add "disc_acc_monitor" to "callbacks_list" if set as "True" in config file
        if make_disc_accuracy_plots:
            if self.config["gan_type"].lower() == "inner":
                data_for_accuracy_test = vertex_level_data_combined
                models_for_accuracy_test = self.cgan
            elif self.config["gan_type"].lower() == "outer":
                data_for_accuracy_test = vertex_s
                models_for_accuracy_test = (self.cgan, self.folding_model.cgan)
            disc_acc_monitor = gm.AccuracyMonitor(
                models_for_accuracy_test,
                output_path,
                frequency=accuracy_frequency,
                training_data=(data_for_accuracy_test, detector_level_data),
                batch_size=self.batch_size,
                noise_dim=self.latent_dim,
                gan_type=self.config["gan_type"].lower(),
            )
            callbacks_list.append(disc_acc_monitor)

        self.cgan.steps_per_epoch = batches_per_epoch
        self.cgan.k = grad_frequency

        # self.cgan.pretrain_outer_generator(labels, ps_labels, 10, 4096)

        history = self.cgan.fit(
            vertex_level_data_combined,
            detector_level_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks_list,
        )

        # The plots of the layer-specific gradients will be made if "make_layer_grad_plots" set as "True" in config file
        if make_layer_grad_plots:
            gradient_monitor.plot_gradients()

        self.history = history.history

        gradient_monitor.analysis()

        return history

    def predict(self, data):
        noise = tf.random.normal(shape=(data.shape[0], self.latent_dim))  # , stddev=10)
        return self.generator.predict([data, noise], batch_size=1024)

    def predict_full(self, data):
        unf_gen_noise = tf.random.normal(shape=(data.shape[0], self.latent_dim))
        unf_gen_output = self.generator.predict([data, unf_gen_noise], batch_size=1024)

        inner_gen_noise = tf.random.normal(
            shape=(unf_gen_output.shape[0], self.latent_dim)
        )
        inner_generated_images = self.cgan.folding_model.static_generator.predict(
            [unf_gen_output, inner_gen_noise], batch_size=1024
        )

        return inner_generated_images

    def predict_graph_mode(self, data):
        noise = tf.random.normal(shape=(tf.shape(data)[0], self.latent_dim))
        return self.static_generator([data, noise], training=False)
