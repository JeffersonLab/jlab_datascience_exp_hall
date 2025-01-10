from aidapt_toolkit.models.tf_cgan_v0 import TF_CGAN_Keras
from aidapt_toolkit.registration import make
import tensorflow as tf
import numpy as np
import yaml

import logging

logger = logging.getLogger("AIDAPT")

#Use to force running in eager mode
#tf.config.run_functions_eagerly(True)

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
        self.unfolding_model.load_inner_GAN(self.unfolding_path)

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
        #real_images, inputs = data
        #print("inputs: ", inputs)
        #print("inputs.shape: ", inputs.shape)
        vertex_s = inputs[:,4]
        #print("vertex_s: ", vertex_s)
        #print("min(vertex_W): ", np.sqrt(np.min(vertex_s)))
        #print("min(vertex_W): ", tf.reduce_min(vertex_s))
        #print("min(vertex_W): ", tf.sqrt(tf.reduce_min(vertex_s)))
        #print("max(vertex_W): ", max(vertex_s))
        #inputs = inputs[:,:-1]
        #real_images = real_images[:,:-1]
        inputs = inputs[:,:4]
        real_images = real_images[:,:4]
        #batch_size = tf.shape(inputs)[0]
        batch_size = tf.shape(vertex_s)[0]
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        #generated_images = self.generator((inputs, noise))

        #generated_images = self.generator((vertex_s, noise))
        
        #zeros_column = tf.zeros((batch_size, 1), dtype=generated_images.dtype)
        #generated_images_padded = tf.concat([generated_images, zeros_column], axis=1)
        #generated_images = self.unfolding_model.predict_graph_mode(generated_images)

        #inner_noise = tf.random.normal(shape=(tf.shape(generated_images)[0], self.unfolding_model.latent_dim))
        #inner_generated_images = self.unfolding_model.static_generator([generated_images, inner_noise], training=False)

        #dummy_noise = tf.zeros_like(noise)
        #inner_generated_images = self.unfolding_model.static_generator([generated_images, dummy_noise], training=False)

        #combined_images = tf.concat([generated_images, real_images], axis=0)

        #combined_images = tf.concat([inner_generated_images, real_images], axis=0)
        #combined_inputs = tf.concat([inputs, inputs], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        #labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
        #print("labels.shape: ", labels.shape)

        # Store the true labels for accuracy calculation before adding noise  
        true_labels = labels
        
        #labels += 0.05*tf.random.uniform(tf.shape(labels))

        disc_updates = 5
        #Train discriminator 
        for _ in range(disc_updates):
            with tf.GradientTape(persistent=True) as disc_tape:
                generated_images = self.generator((vertex_s, noise))
                inner_noise = tf.random.normal(shape=(tf.shape(generated_images)[0], self.unfolding_model.latent_dim))
                inner_generated_images = self.unfolding_model.static_generator([generated_images, inner_noise])#, training=False)
                #print("inner_generated_images.shape: ", inner_generated_images.shape)
                #print("real_images.shape: ", real_images.shape)
                combined_images = tf.concat([inner_generated_images, real_images], axis=0)
                #combined_images = tf.concat([generated_images, real_images], axis=0)
                combined_inputs = tf.concat([inputs, inputs], axis=0)
                predictions = self.discriminator((combined_inputs, combined_images))#, training=True)
                #predictions = self.discriminator(combined_images, training=True)
                d_loss = self.d_loss_fn(labels, predictions)
            discriminator_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(discriminator_gradients, self.discriminator.trainable_weights)
            )
        #print("len(discriminator_gradients): ", len(discriminator_gradients))
        '''
        print("Discriminator trainable weights:")
        for w in self.discriminator.trainable_weights:
            print(w.name, w.shape)
        print("Inner generator trainable weights:")
        for sg in self.unfolding_model.static_generator.trainable_weights:
            print(sg.name, sg.shape)
        print("Generator trainable weights:")
        for w in self.generator.trainable_weights:
            print(w.name, w.shape)
        '''
        # If using logits, apply sigmoid to get probabilities
        if isinstance(self.d_loss_fn, tf.keras.losses.BinaryCrossentropy) and not getattr(self.d_loss_fn, 'from_logits', False):
            #print("No action needed")
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
        #print("discriminator_agg_gradient_norm: ", discriminator_agg_gradient_norm)

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))
        #print("inner_generated_images: ", inner_generated_images)
        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as gen_tape:
            #generated_images = self.generator((inputs, random_latent_vectors))
            generated_images = self.generator((vertex_s, random_latent_vectors))
            #inner_generated_images = self.unfolding_model.predict_graph_mode(generated_images)
            noise = tf.random.normal(shape=(tf.shape(generated_images)[0], self.unfolding_model.latent_dim))
            inner_generated_images = self.unfolding_model.static_generator([generated_images, noise])#, training=False)
            #tf.print("inner_generated_images: ", inner_generated_images)
            #dummy_noise = tf.zeros_like(noise)
            #inner_generated_images = self.unfolding_model.static_generator([generated_images, dummy_noise], training=False)
            #predictions = self.discriminator((inputs, generated_images))#, training=False)
            predictions = self.discriminator((inputs, inner_generated_images))#, training=False)
            g_loss = self.g_loss_fn(misleading_labels, predictions)      
        generator_gradients = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        #for var, grad in zip(self.generator.trainable_weights, generator_gradients):
        #    print(f"Variable: {var.name}, Gradient: {grad}")
        self.g_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_weights))
        #print("len(generator_gradients): ", len(generator_gradients))
        '''
        print("Discriminator trainable weights:")
        for w in self.discriminator.trainable_weights:
            print(w.name, w.shape)
        print("Inner generator trainable weights:")
        for sg in self.unfolding_model.static_generator.trainable_weights:
            print(sg.name, sg.shape)
        print("Generator trainable weights:")
        for w in self.generator.trainable_weights:
            print(w.name, w.shape)
        '''
        gen_layer_grad_norms = []
        for idx, grad in enumerate(generator_gradients):
            norm = tf.norm(grad)
            gen_layer_grad_norms.append(norm)

        # Calculate generator gradient norms
        generator_agg_gradient_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in generator_gradients]))
        #print("generator_agg_gradient_norm: ", generator_agg_gradient_norm)

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
    
    def predict_innerGAN_gen(self, data):
        inputs, real_images = data
        batch_size = tf.shape(inputs)[0]
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        generated_images = self.generator((inputs, noise))
        noise = tf.random.normal(shape=(tf.shape(generated_images)[0], self.unfolding_model.latent_dim))
        inner_generated_images = self.unfolding_model.static_generator([generated_images, noise], training=False)
        return inner_generated_images
