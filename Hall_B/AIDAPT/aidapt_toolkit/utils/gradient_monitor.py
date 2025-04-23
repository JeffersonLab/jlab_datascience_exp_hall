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
        # print("log: ", log)
        # Update training history
        self.history["d_loss"].append(logs["d_loss"])
        self.history["g_loss"].append(logs["g_loss"])
        self.history["discriminator_gradient_norm"].append(
            logs["discriminator_gradient_norm"]
        )
        self.history["generator_gradient_norm"].append(logs["generator_gradient_norm"])

        self.history["Epoch"].append(epoch)

        # Update history for layer-specific gradients
        if self.make_layer_grad_plots:
            # layer_gradients = {}
            for (
                layer_name,
                tracker,
            ) in self.model.discriminator_layer_gradient_norms_tracker.items():
                # for layer in self.model.discriminator_layer_gradient_norms_tracker:
                gradient_norm = tracker.result().numpy()

                if "dense" in layer_name:
                    if not layer_name in self.history:
                        self.history[f"{layer_name}"] = []
                    self.history[f"{layer_name}"].append(gradient_norm)

            for (
                layer_name,
                tracker,
            ) in self.model.generator_layer_gradient_norms_tracker.items():
                gradient_norm = tracker.result().numpy()
                if "dense" in layer_name:
                    if not layer_name in self.history:
                        self.history[f"{layer_name}"] = []
                    self.history[f"{layer_name}"].append(gradient_norm)

    def _save_gradients(self):
        generator_path = os.path.join(self.out_dir, "generator_gradients.npy")
        discriminator_path = os.path.join(self.out_dir, "discriminator_gradients.npy")

        # Save gradients
        np.save(generator_path, self.generator_gradients)
        np.save(discriminator_path, self.discriminator_gradients)

        # Save aggregate gradients
        np.save(
            os.path.join(self.out_dir, "generator_agg_gradients.npy"),
            self.generator_agg_gradients,
        )
        np.save(
            os.path.join(self.out_dir, "discriminator_agg_gradients.npy"),
            self.discriminator_agg_gradients,
        )

    def plot_gradients(self):
        hist = self.history

        # Plot layer-specific gradient norms
        disc_layer_grad_norm_name_arr = []
        gen_layer_grad_norm_name_arr = []
        disc_layer_grad_mean_name_arr = []
        gen_layer_grad_mean_name_arr = []
        for key, value in hist.items():
            if "disc" in key and "dense" in key:
                disc_layer_grad_norm_name_arr.append(key)
            elif "gen" in key and "dense" in key:
                gen_layer_grad_norm_name_arr.append(key)

        # epochs = list(range(0, len(hist[disc_layer_grad_norm_name_arr[1]]) + 1, self.k))
        if len(disc_layer_grad_norm_name_arr) <= 5:
            disc_columns = 1
        if (
            len(disc_layer_grad_norm_name_arr) > 5
            and len(disc_layer_grad_norm_name_arr) <= 10
        ):
            disc_columns = 2
        elif len(disc_layer_grad_norm_name_arr) > 10:
            disc_columns = 3

        # Calculate the number of rows needed based on the number of layers and columns
        disc_rows = (
            len(disc_layer_grad_norm_name_arr) + disc_columns - 1
        ) // disc_columns
        disc_fig_grad_norm_layers, disc_axes_grad_norm_layers = plt.subplots(
            disc_rows, disc_columns, figsize=(disc_columns * 5, disc_rows * 3)
        )
        # disc_fig_grad_norm_layers, disc_axes_grad_norm_layers = plt.subplots(rows, columns, figsize=(10, (len(disc_layer_grad_norm_name_arr) + 1) * 3))

        # Flatten the axes for easy indexing if it's not a 1D array already
        disc_axes_grad_norm_layers = (
            disc_axes_grad_norm_layers.flatten()
            if isinstance(disc_axes_grad_norm_layers, np.ndarray)
            else [disc_axes_grad_norm_layers]
        )

        disc_axes_grad_norm_layers[0].plot(
            np.array(hist[f"discriminator_gradient_norm"])
        )
        disc_axes_grad_norm_layers[0].set_title(f"Discriminator (all layers)")
        disc_axes_grad_norm_layers[0].set_xlabel("Epoch")
        disc_axes_grad_norm_layers[0].set_ylabel("Gradient Norm")
        for i, layer_name in enumerate(disc_layer_grad_norm_name_arr[1:], start=1):
            # disc_axes_grad_norm_layers[i].plot(np.array(hist['Epoch']), np.array(hist[f'{layer_name}']))
            disc_axes_grad_norm_layers[i].plot(
                np.array(hist["Epoch"])[self.k - 1 :: self.k],
                np.array(hist[f"{layer_name}"])[self.k - 1 :: self.k],
            )
            disc_axes_grad_norm_layers[i].set_title(f"Discriminator {layer_name}")
            disc_axes_grad_norm_layers[i].set_xlabel("Epoch")
            disc_axes_grad_norm_layers[i].set_ylabel("Gradient Norm")

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "disc_layer_gradient_norms.png"))
        plt.close(disc_fig_grad_norm_layers)

        if len(gen_layer_grad_norm_name_arr) <= 5:
            gen_columns = 1
        if (
            len(gen_layer_grad_norm_name_arr) > 5
            and len(gen_layer_grad_norm_name_arr) <= 10
        ):
            gen_columns = 2
        elif len(gen_layer_grad_norm_name_arr) > 10:
            gen_columns = 3

        # Calculate the number of rows needed based on the number of layers and columns
        gen_rows = (len(gen_layer_grad_norm_name_arr) + gen_columns - 1) // gen_columns
        gen_fig_grad_norm_layers, gen_axes_grad_norm_layers = plt.subplots(
            gen_rows, gen_columns, figsize=(gen_columns * 5, gen_rows * 3)
        )

        # Flatten the axes for easy indexing if it's not a 1D array already
        gen_axes_grad_norm_layers = (
            gen_axes_grad_norm_layers.flatten()
            if isinstance(gen_axes_grad_norm_layers, np.ndarray)
            else [gen_axes_grad_norm_layers]
        )

        # gen_fig_grad_norm_layers, gen_axes_grad_norm_layers = plt.subplots(len(gen_layer_grad_norm_name_arr), 1, figsize=(10, (len(gen_layer_grad_norm_name_arr) + 1) * 3))
        gen_axes_grad_norm_layers[0].plot(np.array(hist[f"generator_gradient_norm"]))
        gen_axes_grad_norm_layers[0].set_title(f"Generator (all layers)")
        gen_axes_grad_norm_layers[0].set_xlabel("Epoch")
        gen_axes_grad_norm_layers[0].set_ylabel("Gradient Norm")
        for i, layer_name in enumerate(gen_layer_grad_norm_name_arr[1:], start=1):
            gen_axes_grad_norm_layers[i].plot(
                np.array(hist["Epoch"])[self.k - 1 :: self.k],
                np.array(hist[f"{layer_name}"])[self.k - 1 :: self.k],
            )
            gen_axes_grad_norm_layers[i].set_title(f"Generator {layer_name}")
            gen_axes_grad_norm_layers[i].set_xlabel("Epoch")
            gen_axes_grad_norm_layers[i].set_ylabel("Gradient Norm")

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "gen_layer_gradient_norms.png"))
        plt.close(gen_fig_grad_norm_layers)

    def plot_gradients_vs_loss(self):
        hist = self.history

        plt.figure(figsize=(12, 6))
        plt.plot(
            np.array(hist["d_loss"]),
            np.array(hist[f"discriminator_gradient_norm"]),
            label="Discriminator",
        )
        plt.plot(
            np.array(hist["g_loss"]),
            np.array(hist[f"generator_gradient_norm"]),
            label="Generator",
        )
        plt.ylabel("Gradient")
        plt.xlabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "gradient_vs_loss.png"))
        plt.close()

    def analysis(self):

        # Plot training history
        hist = self.history
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.semilogy(np.array(hist["g_loss"]) * 4, label="Generator", alpha=0.75)
        plt.semilogy(np.array(hist["d_loss"]) * 4, label="Discriminator", alpha=0.75)
        # plt.axhline(y=1, linestyle=':', color='k')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        discriminator_gradient_norms = np.array(hist["discriminator_gradient_norm"])
        generator_gradient_norms = np.array(hist["generator_gradient_norm"])

        # Plot gradient norms
        plt.subplot(1, 2, 2)
        plt.plot(generator_gradient_norms, label="Generator", alpha=0.75)
        plt.plot(discriminator_gradient_norms, label="Discriminator", alpha=0.75)
        plt.ylabel("Gradient Norm")
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "training_analysis.png"))
        plt.close()


class ChiSquareMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        training_models,
        data,
        target_distributions,
        output_path,
        latent_dim,
        d_scaler,
        out_dir,
        frequency,
        gan_type,
    ):
        super(ChiSquareMonitor, self).__init__()

        self.data = data
        self.d_scaler = d_scaler
        self.target_distributions = target_distributions
        self.output_path = output_path
        self.frequency = frequency
        self.out_dir = out_dir
        self.chi2_values = []
        self.noise_dim = latent_dim
        self.gan_type = gan_type

        if self.gan_type == "outer":
            self.outer_GAN_model, self.inner_GAN_model = training_models
        elif self.gan_type == "inner":
            self.inner_GAN_model = training_models

    def calculate_chi2(self, generated_distributions, target_distributions, bins=100):
        chi2_values = []
        num_variables = generated_distributions.shape[
            1
        ]  # Assuming multi-dimensional data
        # ranges = [(1,10), (0,4), (-6,0), (0,6.5)]
        ranges = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
        for i in range(num_variables):
            # Bin the generated and target distributions
            gen_hist, bin_edges = np.histogram(
                generated_distributions[:, i], bins=bins, density=True, range=ranges[i]
            )
            target_hist, _ = np.histogram(
                target_distributions[:, i], bins=bins, density=True, range=ranges[i]
            )

            # Avoid division by zero: add a small value to target_hist where it's zero
            target_hist = np.where(target_hist == 0, 1e-6, target_hist)

            # Calculate chi² for this variable
            chi2 = np.sum((gen_hist - target_hist) ** 2 / target_hist)
            reduced_chi2 = chi2 / float(bins)
            chi2_values.append(chi2)

        return chi2_values

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency == 0:
            if self.gan_type == "outer":
                vertex_s = self.data
                vertex_s = tf.convert_to_tensor(vertex_s, dtype=tf.float32)

                noise = tf.random.normal(shape=(vertex_s.shape[0], self.noise_dim))

                # Run prediction to generate GAN's output distribution
                unf_generated_images = self.outer_GAN_model.generator(
                    [vertex_s, noise], training=False
                )
                noise = tf.random.normal(
                    shape=(tf.shape(unf_generated_images)[0], self.noise_dim)
                )
                generated_distributions = self.inner_GAN_model.generator(
                    [unf_generated_images, noise], training=False
                )
            elif self.gan_type == "inner":
                noise = tf.random.normal(shape=(self.data.shape[0], self.noise_dim))

                # Run prediction to generate GAN's output distribution
                generated_distributions = self.inner_GAN_model.generator.predict(
                    [self.data, noise], batch_size=1024
                )

            # Calculate chi^2 between generated and target distributions
            chi2_value = self.calculate_chi2(
                generated_distributions, self.target_distributions
            )
            self.chi2_values.append((epoch, chi2_value))

    def on_train_end(self, logs=None):
        self.plot_chi2_vs_epoch()

    def plot_chi2_vs_epoch(self):
        epochs = [epoch for epoch, chi2_vals in self.chi2_values]
        # Transpose the chi2_values to get individual lists for each distribution
        chi2_per_variable = list(
            zip(*[chi2_vals for epoch, chi2_vals in self.chi2_values])
        )  # Unzip into 4 separate lists

        labels = ["sppim", "spipm", "tpip", "alpha"]

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot each chi² value over the epochs
        for i, chi2_vals in enumerate(chi2_per_variable):
            plt.plot(epochs, chi2_vals, label=f"{labels[i]}")

        # Add plot details
        plt.title(r"$\chi^2$ Test")
        plt.xlabel("Epoch")
        plt.ylabel(r"$\chi^2$")
        plt.yscale("log")
        # plt.ylim(0, 50)
        # plt.grid(True)
        plt.legend()

        plt.savefig(os.path.join(self.out_dir, "chi2_vs_epoch.png"))
        plt.close()

    def get_chi2_history(self):
        return self.chi2_values


class AccuracyMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        training_models,
        out_dir,
        frequency,
        training_data,
        batch_size,
        noise_dim,
        gan_type,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.accuracy_data = []
        self.frequency = frequency
        self.training_data = training_data  # (labels, images)
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.gan_type = gan_type

        if self.gan_type == "inner":
            self.inner_GAN_model = training_models
        elif self.gan_type == "outer":
            self.outer_GAN_model, self.inner_GAN_model = training_models

        self.history = {
            "accuracy": [],
            "Epoch": [],
            "true_labels": [],
            "predicted_scores": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        log = logs or {}
        if (epoch + 1) % self.frequency == 0:
            self.history["Epoch"].append(epoch)

            disc_acc = self.model.disc_acc_tracker.result().numpy()

            # Update history dictionary
            self.history["accuracy"].append(disc_acc)

            # Compute predictions and true labels
            true_labels, predicted_scores = self.compute_predictions()

            # Store data in history
            self.history["true_labels"].extend(true_labels)
            self.history["predicted_scores"].extend(predicted_scores)

    def compute_predictions(self):
        # Unpack training data
        input_data, detector_data = self.training_data
        batch_size = self.batch_size

        # Generate noise
        noise = tf.random.normal([batch_size, self.noise_dim])

        # Create true/false labels for discriminator
        true_labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        if self.gan_type == "outer":
            input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

            # Generate fake images
            unf_generated_images = self.model.generator(
                [input_data[:batch_size], noise], training=False
            )
            noise = tf.random.normal(
                shape=(tf.shape(unf_generated_images)[0], self.noise_dim)
            )
            generated_images = self.inner_GAN_model.generator(
                [unf_generated_images[:batch_size], noise], training=False
            )

            # Get predictions from discriminator
            predictions = self.outer_GAN_model.discriminator(
                [input_data[:batch_size], generated_images], training=False
            )

        elif self.gan_type == "inner":
            input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

            # Generate fake images
            generated_images = self.model.generator(
                [input_data[:batch_size], noise], training=False
            )

            # Get predictions from discriminator
            predictions = self.inner_GAN_model.discriminator(
                [input_data[:batch_size], generated_images], training=False
            )

        # Apply sigmoid if necessary
        if isinstance(
            self.model.d_loss_fn, tf.keras.losses.BinaryCrossentropy
        ) and getattr(self.model.d_loss_fn, "from_logits", False):
            predictions = tf.sigmoid(predictions)

        # Convert tensors to NumPy arrays
        true_labels = true_labels.numpy().flatten()
        predicted_scores = predictions.numpy().flatten()
        # predicted_scores = np.zeros(true_labels)

        return true_labels, predicted_scores

    def on_train_end(self, logs=None):
        self.plot_accuracy_vs_epoch()
        self.plot_roc_curve()

    def plot_accuracy_vs_epoch(self):
        plt.figure(figsize=(10, 6))
        plt.title(r"Discriminator Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel(r"Accuracy")
        plt.plot(self.history["Epoch"], self.history["accuracy"])

        plt.savefig(os.path.join(self.out_dir, "disc_acc_vs_epoch.png"))
        plt.close()

    def plot_roc_curve(self):
        true_labels = np.array(self.history["true_labels"])
        predicted_scores = np.array(self.history["predicted_scores"])

        # Compute ROC curve and ROC area
        fpr, tpr = self.compute_roc_curve(true_labels, predicted_scores)

        # Compute AUC using the trapezoidal rule
        auc_score = np.trapz(tpr, fpr)

        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Discriminator")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.out_dir, "roc_curve.png"))
        plt.close()

    def compute_roc_curve(self, true_labels, predicted_scores):
        # Sort scores and corresponding true labels in descending order
        sorted_indices = np.argsort(-predicted_scores)
        sorted_true_labels = true_labels[sorted_indices]
        sorted_predicted_scores = predicted_scores[sorted_indices]

        # Total number of positive and negative samples
        P = np.sum(sorted_true_labels)
        N = len(sorted_true_labels) - P

        TPR_list = []
        FPR_list = []

        TP = 0
        FP = 0

        # Initialize previous score to a value outside possible score range
        prev_score = -np.inf

        # Loop over all instances
        for i in range(len(sorted_true_labels)):
            score = sorted_predicted_scores[i]
            label = sorted_true_labels[i]

            if score != prev_score:
                # Calculate TPR and FPR
                TPR = TP / P if P > 0 else 0
                FPR = FP / N if N > 0 else 0
                TPR_list.append(TPR)
                FPR_list.append(FPR)
                prev_score = score

            if label == 1:
                TP += 1
            else:
                FP += 1

        # Append the last point
        TPR = TP / P if P > 0 else 0
        FPR = FP / N if N > 0 else 0
        TPR_list.append(TPR)
        FPR_list.append(FPR)

        # Convert lists to NumPy arrays
        FPR_array = np.array(FPR_list)
        TPR_array = np.array(TPR_list)

        return FPR_array, TPR_array
