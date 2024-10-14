from aidapt_toolkit.models.tf_cgan_v0 import TF_CGAN
from aidapt_toolkit.registration import make
import tensorflow as tf
import numpy as np

class TF_OuterGAN_V0(TF_CGAN):
    """ Model for the OuterGAN. 

    
    """
    def __init__(self, config: dict = None, name: str = 'outergan_v0') -> None:
        super().__init__(config, name)

        self.unfolding_path = config['unfolding_path']
        self.unfolding_id = config['unfolding_id']

        self.unfolding_model = make(self.unfolding_id)
        self.unfolding_model.load(self.unfolding_path)

        # This assumes InnerGAN is a TF_CGAN...
        self.unfolding_model.cgan.trainable = False

    def train_step(self, images, labels):
        batch_size = self.batch_size

        valid = tf.ones((batch_size, 1)) + \
            tf.random.normal((batch_size, 1), 0, 0.1)
        fake = tf.zeros((batch_size, 1))

        idx = tf.random.randint(0, images.shape[0], batch_size)
        img_batch, label_batch = images[idx], labels[idx]

        noise = tf.random.normal(0, 1, (batch_size, self.latent_dim))

        gen_imgs = self.generator.predict(
            [label_batch, noise], verbose=0)

        output = self.unfolding_model.predict(gen_imgs)

        gen_imgs = tf.stop_gradient(gen_imgs)

        d_loss_real, acc_real = self.train_discriminator(
            label_batch, img_batch, valid)
        d_loss_fake, acc_fake = self.train_discriminator(
            label_batch, output, fake)
        d_loss_total = (d_loss_real+d_loss_fake)/2

        g_loss = self.train_generator(label_batch, noise, valid)

        return [d_loss_total, d_loss_real,d_loss_fake, acc_real, acc_fake, g_loss]