import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
from jlab_datascience_toolkit.core.jdst_model import JDSTModel
from Hall_B.AIDAPT.utils.config_utils import verify_config

# X is vertex data with shape (N,5), Y is detector data with shape (N,4)
# X is used to condition generator output, Y are the detector values we want to generate
# Generator takes (N, X.shape[1]+latent_dim) while discriminator takes (N, Y.shape[1])


class TF_MLP_GAN_V0(JDSTModel):
    def __init__(self, config, name='unfolding_model_v1') -> None:
        self.module_name = name
        self.required_cfg_keys = ['latent_dim', 'optimizer_lr',
                                  'optimizer_beta1', 'label_shape',
                                  'image_shape', 'generator_layers',
                                  'discriminator_layers', 'batch_size',
                                  'epochs']
        verify_config(config, self.required_cfg_keys)
        self.config = config

        self.latent_dim = self.config[f'latent_dim']
        self.optimizer_lr = self.config[f'optimizer_lr']
        self.optimizer_beta1 = self.config[f'optimizer_beta1']
        self.generator_layers = self.config[f'generator_layers']
        self.discriminator_layers = self.config[f'discriminator_layers']
        self.batch_size = self.config[f'batch_size']
        self.epochs = self.config[f'epochs']

        self.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(
            self.optimizer_lr, self.optimizer_beta1)

        self.generator_optimizer = tf.keras.optimizers.legacy.Adam(
            5*self.optimizer_lr, self.optimizer_beta1)

        self.gen_loss_fn = tf.keras.losses.MeanSquaredError()
        self.disc_loss_fn = tf.keras.losses.MeanSquaredError()
        self.acc_fn = tf.keras.metrics.Accuracy()
        # X-shape
        # self.input_shape = self.config[f'input_shape']
        self.image_shape = self.config[f'image_shape']
        # Y-shape
        # self.output_shape = self.config[f'output_shape']
        self.label_shape = self.config[f'label_shape']

        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(
            loss='mse', optimizer=self.discriminator_optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.summary()

        self.generator.compile(loss='mse', optimizer=self.generator_optimizer)

        # # The generator takes noise as input and vertex_events
        # generator_vertex = tf.keras.layers.Input(shape=self.input_shape)
        # generator_noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        # Det_events = self.generator([generator_vertex, generator_noise])

        # # For the combined model we will only train the generator
        # self.discriminator.trainable = False

        # # The valid takes generated detector_events as input and determines validity
        # valid = self.discriminator([generator_vertex, Det_events])

        # # The combined model  (stacked generator and discriminator)--->Trains generator to fool discriminator
        # self.combined = tf.keras.models.Model(
        #     [generator_vertex, generator_noise], valid)

        # # (!!!) Optimize w.r.t. MSE loss instead of crossentropy
        # # MMD_loss = self.MMD_loss()
        # self.combined.compile(loss='mse', optimizer=self.generator_optimizer)

    def get_info(self):
        print(f'Module name == {self.module_name}')
        print('tf.keras.models.Model will predict in forward() and should be trained prior to use')

    def load(self):
        pass

    def save(self):
        # self.generator.save_weights('GEN_inv_innerGAN_lessarch_withgamma.h5')
        # self.discriminator.save_weights('DES_inv_innerGAN_lessarch_withgamma.h5')
        pass

    def load_config(self):
        return super().load_config()
    
    def save_config(self):
        return super().save_config()
    
    def analysis(self):
        return super().analysis()

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

    # data is [images, labels]
    def train(self, data):
        images, labels = data
        batch_size = self.batch_size
        epochs = self.epochs

        valid = tf.ones((batch_size, 1)) + tf.random.normal((batch_size,1),0,0.1)
        fake = tf.zeros((batch_size, 1))

        pbar = tqdm(range(epochs))
        # d_loss (total, real, fake), #accuracy (real, fake), and #g_loss
        metric_array = np.zeros((epochs, 6))

        for epoch in pbar:
            # ---> Select a random batch of data
            idx = np.random.randint(0, images.shape[0], batch_size)
            img_batch, label_batch = images[idx], labels[idx]

            # --->  Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # --->  Generate a batch of fake detector_events
            gen_imgs = self.generator.predict(
                [label_batch, noise], verbose=0)

            gen_imgs = tf.stop_gradient(gen_imgs)

            d_loss_real, acc_real = self.train_discriminator(label_batch, img_batch, valid)
            d_loss_fake, acc_fake = self.train_discriminator(label_batch, gen_imgs, fake)
            d_loss_total = (d_loss_real+d_loss_fake)/2

            g_loss = self.train_generator(label_batch, noise, valid)
            metric_array[epoch, :] = [d_loss_total, d_loss_real,
                                      d_loss_fake, acc_real, acc_fake, g_loss]

            # pbar.set_postfix_str(f'Metrics: {metric_array[epoch, :]}')
            pbar.set_postfix_str(f'Acc Real/Fake: {acc_real:.3f}/{acc_fake:.3f}, G_loss: {g_loss:.3f}')

        return metric_array

    def predict(self, data):
        noise = np.random.normal(0, 1, (data.shape[0], self.latent_dim))
        return self.generator.predict([data, noise], batch_size=1024)

    def build_generator(self):
        label = tf.keras.layers.Input(shape=(self.label_shape,))
        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        visible = tf.keras.layers.Concatenate()([label, noise])

        x = visible
        for layer in self.generator_layers:
            x = tf.keras.layers.Dense(units=layer.units,
                                      activation=layer.activation, **layer.kwargs)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

        output = tf.keras.layers.Dense(self.image_shape, activation='tanh')(x)

        generator = tf.keras.models.Model(inputs=[label, noise], outputs=[output])
        return generator

    def build_discriminator(self):
        label = tf.keras.layers.Input(shape=(self.label_shape,))
        image = tf.keras.layers.Input(shape=(self.image_shape,))

        x = tf.keras.layers.Concatenate()([label, image])
        for layer in self.discriminator_layers:
            x = tf.keras.layers.Dense(units=layer.units,
                                      activation=layer.activation, **layer.kwargs)(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        discriminator = tf.keras.models.Model(
            inputs=[label, image], outputs=output)
        return discriminator
