from aidapt_toolkit.models.tf_cgan_v0 import TF_CGAN
from aidapt_toolkit.registration import make
import tensorflow as tf
import numpy as np
import yaml

import logging

logger = logging.getLogger("AIDAPT")

class TF_OuterGAN_V0(TF_CGAN):
    """ Model for the OuterGAN. 

    
    """
    def __init__(self, config: dict = None, name: str = 'outergan_v0') -> None:
        super().__init__(config, name)

        self.unfolding_path = config['unfolding_path']
        self.unfolding_id = config['unfolding_id']

        with open(f"{self.unfolding_path}/config.yaml", "r") as f:
            self.unfolding_config = yaml.safe_load(f)

        self.unfolding_model = make(self.unfolding_id, config=self.unfolding_config)
        self.unfolding_model.load(self.unfolding_path)

        # This assumes InnerGAN is a TF_CGAN...
        self.unfolding_model.cgan.trainable = False
    
    def train_step(self, data):
        inputs, real_images = data
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        generated_images = self.generator((inputs, noise))
        generated_images = self.unfolding_model.predict(generated_images)

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
            generated_images = self.generator((inputs, random_latent_vectors))
            generated_images = self.unfolding_model(generated_images)
            predictions = self.discriminator((inputs, generated_images))
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