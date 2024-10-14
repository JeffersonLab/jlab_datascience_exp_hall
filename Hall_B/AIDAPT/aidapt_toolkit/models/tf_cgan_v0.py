import tensorflow as tf
from jlab_datascience_toolkit.core.jdst_model import JDSTModel
from tensorflow.keras.callbacks import Callback
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import inspect
import yaml
import os

def get_optimizer(type: str = 'Adam', optimizer_config: dict = None):
    """Gets a non-legacy Optimizer instance

    Args:
        optimizer_config (dict, optional): Optional configuration dictionary. 
            When not provided, returns an Adam optimizer with default settings.
    Returns:
        tf.keras.optimizer.Optimizer
    """

    if optimizer_config is None:
        optimizer_config = dict()

    module = 'keras.optimizers'
    return tf.keras.optimizers.deserialize(dict(class_name=type, module=module, config=optimizer_config))


def get_layer(type: str = 'Dense', layer_config: dict = None):
    """Gets a non-legacy Layer instance

    Args:
        layer_config (dict): Optional configuration dictionary.
            When not provided, returns a Dense layer with a single unit output and default settings.

    Returns:
        tf.keras.layers.Layer
    """
    if layer_config is None:
        layer_config = dict(units=1)

    module = 'keras.layers'
    return tf.keras.layers.deserialize(dict(class_name=type, module=module, config=layer_config))


def build_sequential_model(layers: list = None):
    if layers is None:
        raise RuntimeError('Cannot build model without a list of layers')

    layer_list = []
    for layer_config in layers:
        # layer_config is a tuple of (type, config_dict)
        type, config_dict = layer_config
        layer_list.append(get_layer(type = type, layer_config=config_dict))
    
    return tf.keras.Sequential(layer_list)

class BatchHisory(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)
        self.model.batch_history = self

class TF_CGAN_Keras(tf.keras.Model):
    def __init__(self, discriminator, generator, noise_dim = 100, batch_size = 32):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.d_loss_tracker = tf.keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = tf.keras.metrics.Mean(name='g_loss')

    def compile(self, d_optimizer, d_loss_fn, g_optimizer, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        inputs, real_images = data
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        generated_images = self.generator((inputs, noise))

        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_inputs = tf.concat([inputs, inputs], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05*tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator((combined_inputs, combined_images))
            d_loss = self.d_loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator((inputs, self.generator((inputs, random_latent_vectors))))
            g_loss = self.g_loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
        }
    
        # If you want to track average losses over an epoch, instead of per batch,
        # use this return call.
        # return {
        #     "d_loss": self.d_loss_tracker.result(),
        #     "g_loss": self.g_loss_tracker.result(),
        # }


class TF_CGAN(JDSTModel):
    def __init__(self, config: dict = None, name: str = 'unfolding_model_v1') -> None:
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
            generator_layers=None,
            discriminator_layers=None,
            epochs=1,
            batch_size=32,
            random_seed=42)
        
        if config is not None:
            self.config.update(config)

        self.set_model_variables()

        self.rng = tf.random.Generator.from_seed(self.config['random_seed'])

        # TODO: Make sure random_seed also controls model weights/optimizers/etc

        self.discriminator_optimizer = get_optimizer(
            *self.config['discriminator_optimizer'])

        self.generator_optimizer = get_optimizer(
            *self.config['generator_optimizer'])

        #TODO: Make this configurable in the config
        self.gen_loss_fn = tf.keras.losses.MeanSquaredError()
        self.disc_loss_fn = tf.keras.losses.MeanSquaredError()
        self.acc_fn = tf.keras.metrics.binary_accuracy

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.cgan = TF_CGAN_Keras(self.discriminator, self.generator, 
                            noise_dim=self.config['latent_dim'], 
                            batch_size=self.config['batch_size'])

        self.cgan.compile(self.discriminator_optimizer, self.disc_loss_fn, 
                          self.generator_optimizer, self.gen_loss_fn)

        # self.discriminator.compile(loss='mse', 
        #                            optimizer=self.discriminator_optimizer, 
        #                            metrics=['accuracy'])
        # self.generator.compile(loss='mse', optimizer=self.generator_optimizer)

    def set_model_variables(self):
        self.latent_dim = self.config[f'latent_dim']
        self.generator_layers = self.config[f'generator_layers']
        self.discriminator_layers = self.config[f'discriminator_layers']
        self.batch_size = self.config[f'batch_size']
        self.epochs = self.config[f'epochs']
        self.image_shape = self.config[f'image_shape']
        self.label_shape = self.config[f'label_shape']

    def build_generator(self):
        label = tf.keras.layers.Input(shape=(self.label_shape,))
        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Concatenate()([label, noise])

        model = build_sequential_model(self.config['generator_layers'])
        
        x = model(x)

        output = tf.keras.layers.Dense(self.image_shape, activation='tanh')(x)

        generator = tf.keras.models.Model(
            inputs=[label, noise], outputs=[output])
        return generator

    def build_discriminator(self):
        label = tf.keras.layers.Input(shape=(self.label_shape,))
        image = tf.keras.layers.Input(shape=(self.image_shape,))

        x = tf.keras.layers.Concatenate()([label, image])

        model = build_sequential_model(self.config['discriminator_layers'])
        x = model(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        discriminator = tf.keras.models.Model(
            inputs=[label, image], outputs=output)
        return discriminator

    def get_info(self):
        print(inspect.getdoc(self))

    def load_config(self):
        return super().load_config()

    def save_config(self):
        return super().save_config()

    def load(self, filepath):
        # Set all configuration settings
        with open(os.path.join(filepath, 'config.yaml'), 'r') as f:
            self.config.update(yaml.safe_load(f))
        self.set_model_variables()

        # Build correct network before loading weights
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Load network weights
        self.generator.load_weights(os.path.join(filepath, 'generator.weights.h5'))
        self.discriminator.load_weights(os.path.join(filepath, 'discriminator.weights.h5'))

        self.cgan = TF_CGAN_Keras(self.discriminator, self.generator, 
                            noise_dim=self.config['latent_dim'], 
                            batch_size=self.config['batch_size'])

        self.cgan.compile(self.discriminator_optimizer, self.disc_loss_fn, 
                          self.generator_optimizer, self.gen_loss_fn)
        # TODO load optimizers specified in config...

    def save(self, filepath):
        os.makedirs(filepath)
        self.generator.save_weights(os.path.join(filepath, 'generator.weights.h5'))
        self.discriminator.save_weights(os.path.join(filepath, 'discriminator.weights.h5'))
        with open(os.path.join(filepath, 'config.yaml'), 'w') as f:
            #yaml.safe_dump(self.config, f)
            OmegaConf.save(self.config, f)
        # TODO: Save state of optimizers

    def analysis(self, path=None):
        # Plot training history
        hist = self.cgan.batch_history.history

        plt.semilogy(np.array(hist['g_loss'])*4, label='Generator')
        plt.semilogy(np.array(hist['d_loss'])*4, label='Discriminator')
        # plt.semilogy(self.history['d_loss_fake']*4, label='Discriminator Fake')
        plt.axhline(y=1, linestyle=':', color='k')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(path, 'model_history.png'))

    def train(self, data):
        images, labels = data
        history = self.cgan.fit(labels, images, batch_size = self.batch_size, epochs = self.epochs, callbacks=[BatchHisory()])
        self.history = history.history
        return history

    def predict(self, data):
        noise = tf.random.normal(shape=(data.shape[0], self.latent_dim))
        return self.generator.predict([data, noise], batch_size=1024)
