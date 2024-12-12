from aidapt_toolkit.models.tf_cgan_v0 import TF_CGAN_Keras
from aidapt_toolkit.registration import make
import tensorflow as tf
import numpy as np
import yaml

import logging

logger = logging.getLogger("AIDAPT")

class TF_OuterGAN_V0(TF_CGAN_Keras):
    """ Model for the OuterGAN. 

    
    """
    def __init__(self, unfolding_path, unfolding_id, discriminator, generator, noise_dim = 100, batch_size = 32) -> None:
        super().__init__(discriminator, generator, noise_dim = 100, batch_size = 32)

        self.unfolding_path = unfolding_path
        self.unfolding_id = unfolding_id

        with open(f"{self.unfolding_path}/config.yaml", "r") as f:
            self.unfolding_config = yaml.safe_load(f)

        self.unfolding_model = make(self.unfolding_id, config=self.unfolding_config)
        self.unfolding_model.load(self.unfolding_path)

        # This assumes InnerGAN is a TF_CGAN...
        self.unfolding_model.cgan.trainable = False

        # Initialize loss and overall gradient trackers
        self.d_loss_tracker = tf.keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = tf.keras.metrics.Mean(name='g_loss')
        self.discriminator_gradient_norm_tracker = tf.keras.metrics.Mean(name='discriminator_gradient_norm')
        self.generator_gradient_norm_tracker = tf.keras.metrics.Mean(name='generator_gradient_norm')

        # Initialize discriminator accuracy trackers 
        self.disc_acc_tracker = tf.keras.metrics.Mean(name='disc_acc')
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
                        #print("sub_layer.path: ", sub_layer.path)
                        self.discriminator_layer_gradient_norms_tracker[f'disc_{sub_layer.path}'] = tf.keras.metrics.Mean(name=f'disc_{sub_layer.path}_grad_norm')
                else:
                    for sub_layer in layer.trainable_weights:
                        self.discriminator_layer_gradient_norms_tracker[f'disc_{sub_layer.path}'] = tf.keras.metrics.Mean(name=f'disc_{sub_layer.path}_grad_norm')

        # Initialize generator gradient norm trackers for each layer 
        self.generator_layer_gradient_norms_tracker = {}
        for layer in generator.layers:
            if layer.trainable_weights:
                if isinstance(layer, tf.keras.Sequential):
                    for sub_layer in layer.trainable_weights:
                        self.generator_layer_gradient_norms_tracker[f'gen_{sub_layer.path}'] = tf.keras.metrics.Mean(name=f'gen_{sub_layer.path}_grad_norm')

                else:
                    for sub_layer in layer.trainable_weights:
                        self.generator_layer_gradient_norms_tracker[f'gen_{sub_layer.path}'] = tf.keras.metrics.Mean(name=f'gen_{sub_layer.path}_grad_norm')
    
    def train_step(self, data):
        inputs, real_images = data
        #batch_size = tf.shape(real_images)[0]
        batch_size = tf.shape(inputs)[0]
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        #print("inputs.shape: ", inputs.shape)
        #print("real_images.shape: ", real_images.shape)
        #print("noise.shape: ", noise.shape)
        generated_images = self.generator((inputs, noise))
        #print("generated_images.shape: ", generated_images.shape)
        zeros_column = tf.zeros((batch_size, 1), dtype=generated_images.dtype)
        generated_images_padded = tf.concat([generated_images, zeros_column], axis=1)
        #print("generated_images_padded.shape: ", generated_images_padded.shape)
        #generated_images = self.unfolding_model.predict_graph_mode(generated_images_padded)
        generated_images = self.unfolding_model.predict_graph_mode(generated_images)
        #generated_images = self.unfolding_model.generator([generated_images, noise], training=False)

        #tf.print("generated_images: ", generated_images)

        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_inputs = tf.concat([inputs, inputs], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # Store the true labels for accuracy calculation before adding noise  
        true_labels = labels
        
        labels += 0.05*tf.random.uniform(tf.shape(labels))

        disc_updates = 5
        #Train discriminator 
        for _ in range(disc_updates):
            with tf.GradientTape(persistent=True) as tape:
                predictions = self.discriminator((combined_inputs, combined_images))
                d_loss = self.d_loss_fn(labels, predictions)
            discriminator_gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(discriminator_gradients, self.discriminator.trainable_weights)
            )

        # If using logits, apply sigmoid to get probabilities
        if isinstance(self.d_loss_fn, tf.keras.losses.BinaryCrossentropy) and not getattr(self.d_loss_fn, 'from_logits', False):
            pass  # No action needed
        else:
            # Apply sigmoid if predictions are logits
            predictions = tf.sigmoid(predictions)

        predicted_scores = predictions
        predicted_labels = tf.cast(predictions > 0.5, tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, tf.round(true_labels)), tf.float32))

        disc_layer_grad_norms = []
        for idx, grad in enumerate(discriminator_gradients):
            norm = tf.norm(grad)
            disc_layer_grad_norms.append(norm)

        # Calculate discriminator gradient norms
        discriminator_agg_gradient_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in discriminator_gradients]))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            generated_images = self.generator((inputs, random_latent_vectors))
            #generated_images = self.unfolding_model.generator(generated_images)
            #generated_images = self.unfolding_model.generator([inputs, noise])#, training=False)
            #zeros_column = tf.zeros((batch_size, 1), dtype=generated_images.dtype)
            #generated_images_padded = tf.concat([generated_images, zeros_column], axis=1)
            #generated_images = self.unfolding_model.predict_graph_mode(generated_images_padded)
            generated_images = self.unfolding_model.predict_graph_mode(generated_images)
            predictions = self.discriminator((inputs, generated_images))
            #print("predictions: ", predictions)
            #predictions = self.discriminator((inputs, self.generator((inputs, random_latent_vectors))))
            g_loss = self.g_loss_fn(misleading_labels, predictions)
            #print("g_loss: ", g_loss)
            #print("predictions: ", predictions)
        generator_gradients = tape.gradient(g_loss, self.generator.trainable_weights)
        #print("grads: ", grads)
        self.g_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_weights))

        gen_layer_grad_norms = []
        for idx, grad in enumerate(generator_gradients):
            norm = tf.norm(grad)
            gen_layer_grad_norms.append(norm)

        # Calculate generator gradient norms
        generator_agg_gradient_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in generator_gradients]))

        # ----------------------------- Update tracked metrics ------------------------------- #
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.discriminator_gradient_norm_tracker.update_state(discriminator_agg_gradient_norm)
        self.generator_gradient_norm_tracker.update_state(generator_agg_gradient_norm)

        #Layer-specfic gradient norms for discriminator
        disc_item_counter = 0
        for item in self.discriminator_layer_gradient_norms_tracker:
            self.discriminator_layer_gradient_norms_tracker[item].update_state(disc_layer_grad_norms[disc_item_counter])
            disc_item_counter += 1
            #print("item: ", item)

        #Layer-specific gradient norms for generator
        gen_item_counter = 0
        for item in self.generator_layer_gradient_norms_tracker:
            self.generator_layer_gradient_norms_tracker[item].update_state(gen_layer_grad_norms[gen_item_counter])
            gen_item_counter += 1

        self.disc_acc_tracker.update_state(accuracy)
        self.true_labels_list.append(true_labels)
        self.predicted_scores_list.append(predictions)

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "discriminator_gradient_norm": self.discriminator_gradient_norm_tracker.result(),
            "generator_gradient_norm": self.generator_gradient_norm_tracker.result(),
        }
        # ------------------------------------------------------------------------------------ #
    

