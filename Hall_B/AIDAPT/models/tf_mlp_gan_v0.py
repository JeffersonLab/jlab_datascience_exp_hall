import tensorflow as tf
import numpy as np
from jlab_datascience_toolkit.core.jdst_model import JDSTModel
from Hall_B.AIDAPT.utils.config_utils import verify_config
import matplotlib.pyplot as plt
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

class TF_MLP_GAN_V0(JDSTModel):
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
        self.acc_fn = tf.keras.metrics.Accuracy()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='mse', optimizer=self.discriminator_optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='mse', optimizer=self.generator_optimizer)

    def set_model_variables(self):
        self.latent_dim = self.config[f'latent_dim']
        self.generator_layers = self.config[f'generator_layers']
        self.discriminator_layers = self.config[f'discriminator_layers']
        self.batch_size = self.config[f'batch_size']
        self.epochs = self.config[f'epochs']
        self.image_shape = self.config[f'image_shape']
        self.label_shape = self.config[f'label_shape']

    def get_info(self):
        print(inspect.getdoc(self))

    def load(self, filepath):
        # Set all configuration settings
        with open(os.path.join(filepath, 'config.yaml'), 'r') as f:
            self.config.update(yaml.safe_load(f))
        self.set_model_variables()

        # Build correct network before loading weights
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Load network weights
        self.generator.load_weights(os.path.join(filepath, 'generator_weights'))
        self.discriminator.load_weights(os.path.join(filepath, 'discriminator_weights'))

        # TODO load optimizers specified in config...

    def save(self, filepath):
        self.generator.save_weights(os.path.join(filepath, 'generator_weights'))
        self.discriminator.save_weights(os.path.join(filepath, 'discriminator_weights'))
        with open(os.path.join(filepath, 'config.yaml'), 'w') as f:
            yaml.safe_dump(self.config, f)
        # TODO: Save state of optimizers

    def load_config(self):
        return super().load_config()

    def save_config(self):
        return super().save_config()

    def analysis(self):
        # Plot training history
        plt.semilogy(self.history['g_loss']*4, label='Generator')
        plt.semilogy(self.history['d_loss_real']*4, label='Discriminator Real')
        plt.semilogy(self.history['d_loss_fake']*4, label='Discriminator Fake')
        plt.axhline(y=1)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show() 

    def train(self, data):
        images, labels = data
        epochs = self.epochs
        metric_names = ['d_loss_total', 'd_loss_real', 'd_loss_fake', 'acc_real', 'acc_fake', 'g_loss']

        callbacks = tf.keras.callbacks.CallbackList(
            add_progbar=True, epochs=1, steps=epochs, 
            verbose=1, stateful_metrics=metric_names)

        metric_array = np.zeros((epochs, 6))
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        for epoch in range(epochs):
            callbacks.on_train_batch_begin(epoch)
            metrics = self.train_step(images, labels)
            callbacks.on_train_batch_end(epoch, metrics['history'])
            metric_array[epoch, :] = list(metrics['history'].values())

        callbacks.on_epoch_end(0, {})
        callbacks.on_train_end({})

        self.history = {}
        for metric, name in zip(metric_array.T, metric_names):
            self.history[name] = metric
        return self.history

    def predict(self, data):
        noise = np.random.normal(0, 1, (data.shape[0], self.latent_dim))
        return self.generator.predict([data, noise], batch_size=1024)

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

    def train_step(self, images, labels):
        batch_size = self.batch_size

        noise = self.rng.normal((batch_size, 1), 0, 0.1)
        valid = tf.ones((batch_size, 1)) + noise
        fake = tf.zeros((batch_size, 1))
        idx = np.random.randint(0, images.shape[0], batch_size)
        img_batch, label_batch = images[idx], labels[idx]

        # --->  Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        # --->  Generate a batch of fake detector_events
        gen_imgs = self.generator.predict(
            [label_batch, noise], verbose=0)

        gen_imgs = tf.stop_gradient(gen_imgs)

        d_loss_real, acc_real = self.train_discriminator(
            label_batch, img_batch, valid)
        d_loss_fake, acc_fake = self.train_discriminator(
            label_batch, gen_imgs, fake)
        d_loss_total = (d_loss_real+d_loss_fake)/2

        g_loss = self.train_generator(label_batch, noise, valid)

        metrics = [d_loss_total, d_loss_real, d_loss_fake, acc_real, acc_fake, g_loss]
        metric_names = ['d_loss_total', 'd_loss_real', 'd_loss_fake', 'acc_real', 'acc_fake', 'g_loss']
        history = {'history': {name: metric for name, metric in zip(metric_names, metrics)}}
        return history


    @tf.function
    def train_generator(self, labels, noise, valid):
        with tf.GradientTape() as tape:
            fake_images = self.generator([labels, noise])
            predictions = self.discriminator([labels, fake_images])
            g_loss = self.gen_loss_fn(valid, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        return g_loss

    @tf.function
    def train_discriminator(self, labels, images, valid):
        with tf.GradientTape() as tape:
            predictions = self.discriminator([labels, images])
            d_loss = self.disc_loss_fn(valid, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))
        acc = self.acc_fn(valid, predictions)
        return d_loss, acc
