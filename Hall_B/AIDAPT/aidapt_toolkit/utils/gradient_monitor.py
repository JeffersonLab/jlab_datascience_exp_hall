import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class GradientMonitor(tf.keras.callbacks.Callback):
    def __init__(self, input_model, out_dir, k, make_layer_grad_plots):
        super().__init__()
        self.input_model = input_model
        self.out_dir = out_dir
        self.k = k
        self.gradient_data = {}
        self.make_layer_grad_plots = make_layer_grad_plots

        # Ensure the output directory exists
        os.makedirs(out_dir, exist_ok=True)

        # Initialize dictionary for training history
        self.history = {
            "d_loss": [],
            "g_loss": [],
            "discriminator_gradient_norm": [],
            "generator_gradient_norm": [],
            "Epoch": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        log = logs or {}
        #print("log: ", log)
        # Update training history
        self.history["d_loss"].append(logs["d_loss"])
        self.history["g_loss"].append(logs["g_loss"])
        self.history["discriminator_gradient_norm"].append(logs["discriminator_gradient_norm"])
        self.history["generator_gradient_norm"].append(logs["generator_gradient_norm"])

        self.history["Epoch"].append(epoch)

        # Update history for layer-specific gradients
        if (self.make_layer_grad_plots):
            #layer_gradients = {}
            for layer_name, tracker in self.model.discriminator_layer_gradient_norms_tracker.items():
            #for layer in self.model.discriminator_layer_gradient_norms_tracker:
                gradient_norm = tracker.result().numpy()
                #print("layer_name: ", layer_name)
                #print("gradient_norm: ", gradient_norm)
                
                if 'dense' in layer_name:
                    if not layer_name in self.history:
                        self.history[f"{layer_name}"] = []
                    self.history[f"{layer_name}"].append(gradient_norm)

            for layer_name, tracker in self.model.generator_layer_gradient_norms_tracker.items():
                gradient_norm = tracker.result().numpy()
                if 'dense' in layer_name:
                    if not layer_name in self.history:
                        self.history[f"{layer_name}"] = []
                    self.history[f"{layer_name}"].append(gradient_norm)
                    
        # Save and plot gradients every k epochs
        #if (epoch+1) % self.k == 0:
        #    print("Plotting gradients")
        #    #self.plot_gradients()
        #    #self._save_gradients()
            
    def _save_gradients(self):
        generator_path = os.path.join(self.out_dir, 'generator_gradients.npy')
        discriminator_path = os.path.join(self.out_dir, 'discriminator_gradients.npy')

        # Save gradients
        np.save(generator_path, self.generator_gradients)
        np.save(discriminator_path, self.discriminator_gradients)

        # Save aggregate gradients
        np.save(os.path.join(self.out_dir, 'generator_agg_gradients.npy'), self.generator_agg_gradients)
        np.save(os.path.join(self.out_dir, 'discriminator_agg_gradients.npy'), self.discriminator_agg_gradients)

    def plot_gradients(self):
        hist = self.history

        print("hist.items(): ", hist.items())

        # Generate x-axis values (epochs) based on k
        #epochs = list(range(0, len(next(iter(self.generator_gradients.values())))))
        #print(epochs)

        # Plot layer-specific gradient norms
        disc_layer_grad_norm_name_arr = []
        gen_layer_grad_norm_name_arr = []
        disc_layer_grad_mean_name_arr = []
        gen_layer_grad_mean_name_arr = []
        for key, value in hist.items():
            if 'disc' in key and 'dense' in key:
                disc_layer_grad_norm_name_arr.append(key)
            elif 'gen' in key and 'dense' in key:
                gen_layer_grad_norm_name_arr.append(key)

        #epochs = list(range(0, len(hist[disc_layer_grad_norm_name_arr[1]]) + 1, self.k))
        
        print("disc_layer_grad_norm_name_arr: ", disc_layer_grad_norm_name_arr)
        disc_fig_grad_norm_layers, disc_axes_grad_norm_layers = plt.subplots(len(disc_layer_grad_norm_name_arr), 1, figsize=(10, (len(disc_layer_grad_norm_name_arr) + 1) * 3))
        disc_axes_grad_norm_layers[0].plot(np.array(hist[f'discriminator_gradient_norm']))
        disc_axes_grad_norm_layers[0].set_title(f'Discriminator (all layers)')
        disc_axes_grad_norm_layers[0].set_xlabel('Epoch')
        disc_axes_grad_norm_layers[0].set_ylabel('Gradient Norm')
        for i, layer_name in enumerate(disc_layer_grad_norm_name_arr[1:], start=1):
            #disc_axes_grad_norm_layers[i].plot(np.array(hist['Epoch']), np.array(hist[f'{layer_name}']))
            disc_axes_grad_norm_layers[i].plot(np.array(hist['Epoch'])[self.k-1::self.k], np.array(hist[f'{layer_name}'])[self.k-1::self.k])
            disc_axes_grad_norm_layers[i].set_title(f'Discriminator {layer_name}')
            disc_axes_grad_norm_layers[i].set_xlabel('Epoch')
            disc_axes_grad_norm_layers[i].set_ylabel('Gradient Norm')

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'disc_layer_gradient_norms.png'))
        plt.close(disc_fig_grad_norm_layers)
        
        gen_fig_grad_norm_layers, gen_axes_grad_norm_layers = plt.subplots(len(gen_layer_grad_norm_name_arr), 1, figsize=(10, (len(gen_layer_grad_norm_name_arr) + 1) * 3))
        gen_axes_grad_norm_layers[0].plot(np.array(hist[f'generator_gradient_norm']))
        gen_axes_grad_norm_layers[0].set_title(f'Generator (all layers)')
        gen_axes_grad_norm_layers[0].set_xlabel('Epoch')
        gen_axes_grad_norm_layers[0].set_ylabel('Gradient Norm')
        for i, layer_name in enumerate(gen_layer_grad_norm_name_arr[1:], start=1):
            gen_axes_grad_norm_layers[i].plot(np.array(hist['Epoch'])[self.k-1::self.k], np.array(hist[f'{layer_name}'])[self.k-1::self.k])
            gen_axes_grad_norm_layers[i].set_title(f'Generator {layer_name}')
            gen_axes_grad_norm_layers[i].set_xlabel('Epoch')
            gen_axes_grad_norm_layers[i].set_ylabel('Gradient Norm')

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'gen_layer_gradient_norms.png'))
        plt.close(gen_fig_grad_norm_layers)
        

    def plot_gradients_vs_loss(self):
        hist = self.history

        plt.figure(figsize=(12, 6))
        plt.plot(np.array(hist['d_loss']), np.array(hist[f'discriminator_gradient_norm']), label='Discriminator')
        plt.plot(np.array(hist['g_loss']), np.array(hist[f'generator_gradient_norm']), label='Generator')
        plt.ylabel('Gradient')
        plt.xlabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'gradient_vs_loss.png'))
        plt.close()
        
        
    def analysis(self):
        #print("self: ", self)
        # Plot training history
        hist = self.history
        #print("hist: ", hist)
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.semilogy(np.array(hist['g_loss'])*4, label='Generator')
        plt.semilogy(np.array(hist['d_loss'])*4, label='Discriminator')
        #plt.axhline(y=1, linestyle=':', color='k')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        discriminator_gradient_norms = np.array(hist['discriminator_gradient_norm'])
        generator_gradient_norms = np.array(hist['generator_gradient_norm'])
        #print(generator_gradient_norms)
        
        # Plot gradient norms
        plt.subplot(1, 2, 2)
        plt.plot(discriminator_gradient_norms, label='Discriminator', alpha=0.75)
        plt.plot(generator_gradient_norms, label='Generator', alpha=0.75)
        plt.ylabel('Gradient Norm')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'training_analysis.png'))
        plt.close()


class ChiSquareMonitor(tf.keras.callbacks.Callback):
    def __init__(self, monitored_model, data, target_distributions, output_path, latent_dim, d_scaler, out_dir, frequency):
        super(ChiSquareMonitor, self).__init__()
        self.monitored_model = monitored_model
        self.data = data
        self.d_scaler = d_scaler
        self.target_distributions = target_distributions
        self.output_path = output_path
        #self.chi2_func = chi2_func
        self.frequency = frequency
        self.out_dir = out_dir
        self.chi2_values = []
        self.noise_dim = latent_dim

    def calculate_chi2(self, generated_distributions, target_distributions, bins=100):
        #chi2 = np.sum((target_distributions - generated_distributions)**2 / target_distributions)
        chi2_values = []
        num_variables = generated_distributions.shape[1]  # Assuming multi-dimensional data
        ranges = [(1,10), (0,4), (-6,0), (0,6.5)]
        for i in range(num_variables):
            # Bin the generated and target distributions
            gen_hist, bin_edges = np.histogram(generated_distributions[:, i], bins=bins, density=True, range=ranges[i])
            target_hist, _ = np.histogram(target_distributions[:, i], bins=bins, density=True, range=ranges[i])
            
            # Avoid division by zero: add a small value to target_hist where it's zero
            target_hist = np.where(target_hist == 0, 1e-6, target_hist)

            #print("gen_hist: ", gen_hist)
            #print("target_hist: ", target_hist)
            #print("bins: ", bins)
            # Calculate chi² for this variable
            chi2 = np.sum((gen_hist - target_hist)**2 / target_hist)
            reduced_chi2 = chi2 / float(bins)
            chi2_values.append(chi2)

        return chi2_values
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency == 0:
            noise = tf.random.normal(shape=(self.data.shape[0], self.noise_dim))

            # Run prediction to generate GAN's output distribution
            generated_distributions = self.monitored_model.generator.predict([self.data, noise], batch_size=1024)
            generated_distributions = self.d_scaler.reverse(generated_distributions)
            #print("generated_distributions: ", generated_distributions)
            #print("self.target_distributions: ", self.target_distributions)

            # Calculate chi^2 between generated and target distributions
            chi2_value = self.calculate_chi2(generated_distributions, self.target_distributions)
            self.chi2_values.append((epoch, chi2_value))

            #num_data_points = np.prod(self.target_distributions.shape)
            #print("num_data_points: ", num_data_points)
            #reduced_chi2 = chi2 / num_data_points
            
            #print(f'Epoch {epoch+1}: reduced Chi^2 value: {chi2_value}')

    def on_train_end(self, logs=None):
        self.plot_chi2_vs_epoch()
    
    def plot_chi2_vs_epoch(self):
        epochs = [epoch for epoch, chi2_vals in self.chi2_values]
        # Transpose the chi2_values to get individual lists for each distribution
        chi2_per_variable = list(zip(*[chi2_vals for epoch, chi2_vals in self.chi2_values]))  # Unzip into 4 separate lists

        labels = ['sppim', 'spipm', 'tpip', 'alpha']
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot each chi² value over the epochs
        for i, chi2_vals in enumerate(chi2_per_variable):
            plt.plot(epochs, chi2_vals, label=f'{labels[i]}')
            
        # Add plot details
        plt.title(r'$\chi^2$ Test')
        plt.xlabel('Epoch')
        plt.ylabel(r'$\chi^2$')
        plt.yscale("log")
        #plt.ylim(0, 50)
        #plt.grid(True)
        plt.legend()

        plt.savefig(os.path.join(self.out_dir, 'chi2_vs_epoch.png'))
        plt.close()

    def get_chi2_history(self):
        return self.chi2_values

class AccuracyMonitor(tf.keras.callbacks.Callback):
    def __init__(self, input_model, out_dir, frequency):
        super().__init__()
        self.input_model = input_model
        self.out_dir = out_dir
        #self.k = k
        self.accuracy_data = []
        self.frequency = frequency
        #self.make_layer_grad_plots = make_layer_grad_plots
        #super(AccuracyMonitor, self).__init__()

        self.history = {
            "accuracy": [],
            "Epoch": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        log = logs or {}
        if (epoch + 1) % self.frequency == 0:
            #self.history["accuracy"].append(logs["disc_acc"])
            self.history["Epoch"].append(epoch)

            disc_acc = self.model.disc_acc_tracker.result().numpy()
            #print(f"Epoch {epoch}: Discriminator Accuracy: {disc_acc}")
        
            #print("self.history['accuracy']: ", self.history["accuracy"])
            #avg_accuracy = np.mean(self.input_model.disc_acc_tracker)
            #self.model.epoch_accuracy.append(avg_accuracy)
            #print(f"Epoch {epoch + 1}, Discriminator Accuracy: {avg_accuracy:.4f}")

            # Update history dictionary
            self.history["accuracy"].append(disc_acc)
        
            # Reset the batch accuracy list for the next epoch
            #self.model.disc_acc_tracker = []

    def on_train_end(self, logs=None):
        self.plot_accuracy_vs_epoch()

    def plot_accuracy_vs_epoch(self):
        plt.figure(figsize=(10, 6))
        plt.title(r'Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel(r'Accuracy')
        plt.plot(self.history["Epoch"], self.history["accuracy"])

        plt.savefig(os.path.join(self.out_dir, 'disc_acc_vs_epoch.png'))
        plt.close()
