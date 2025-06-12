import tensorflow as tf
from tensorflow import math
import keras
import numpy as np
from keras import Model, layers


class SceneContentApproximator(Model):



    def __init__(self, num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda, **kwargs):
        super().__init__(**kwargs)


        self.num_kernels = num_kernels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.learning_rate = learning_rate
        self.loss_constant_alpha = loss_constant_alpha
        self.loss_constant_lambda = loss_constant_lambda
        self.conv = layers.Conv2D(
                                    num_kernels, 
                                    (kernel_height,kernel_width),
                                    use_bias= False, 
                                    kernel_regularizer = KernelDiversityLoss(loss_constant_alpha, loss_constant_lambda, num_kernels, kernel_height, kernel_width),
                                    kernel_constraint = KernelConstraint((kernel_height, kernel_height, 1, num_kernels)), # num_colorchannels = 1 for grayscaled images
                                    padding='same' # to let the ouput of the layer have the same size as the input
                                    )
        
    def call(self, input):
        return self.conv(input)
    
    # expects grayscaled images (num_colorchannels = 1)
    def train(self, dataset, epochs):

        num_batches = 0
        for _ in dataset:
            num_batches += 1

        history = {
            'recon_loss': [],
            'kernel_diversity_loss': [],
            'total_loss': []
        }

        def custom_loss(y_true, y_pred):
            # y_true.shape (batch_size, resize_height, resize_width, num_colorchannels)
            # y_true.shape (batch_size, resize_height, resize_width, num_kernels)

            # y_true gets broadcasted (num_colorchannels = 1 required)
            return  math.reduce_sum(math.square(y_true - y_pred))
                                                
        optimizer = keras.optimizers.AdamW(self.learning_rate)

        def train_step(image_batch):
            with tf.GradientTape() as tape:

                # shape: ()
                prediction = self(image_batch, training=True)

                recon_loss = custom_loss(image_batch, prediction)
                diversity_loss = self.losses[0]  # KernelDiversityLoss

                # single value
                total_loss = recon_loss + diversity_loss

            # Get gradients and update weights
            gradients = tape.gradient(total_loss, self.trainable_variables)
            # Check if any gradient is None
            if any(grad is None for grad in gradients):
                tf.print("Warning: One or more gradients are None.")
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return total_loss, recon_loss, diversity_loss


        
        for i in range (epochs):
            tf.print("Epoch: ", i)
            epoch_recon_loss = 0.0
            epoch_kernel_diversity_loss = 0.0
            epoch_total_loss = 0.0
            for image_batch in dataset:
                total_loss, recon_loss, diversity_loss = train_step(image_batch)
                epoch_recon_loss += recon_loss
                epoch_kernel_diversity_loss += diversity_loss
                epoch_total_loss += total_loss

            history['recon_loss'].append(epoch_recon_loss / num_batches)
            history['kernel_diversity_loss'].append(epoch_kernel_diversity_loss / num_batches)
            history['total_loss'].append(epoch_total_loss / num_batches)

        return history

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_kernels': self.num_kernels,
            'kernel_height': self.kernel_height,
            'kernel_width': self.kernel_width,
            'learning_rate': self.learning_rate,
            'loss_constant_alpha': self.loss_constant_alpha,
            'loss_constant_lambda': self.loss_constant_lambda,
            # ...add any other args...
        })
        return config
    


class KernelConstraint(keras.constraints.Constraint):
    def __init__(self, kernel_shape):
        super().__init__()

        kernel_height, kernel_width, num_colorchannels, num_kernels = kernel_shape

        mask = np.ones(kernel_shape)
        mask[kernel_height//2, kernel_width//2, :, :] = 0
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    def __call__(self, w):

        w_zeroed = w * self.mask

        # Compute sum of each kernel
        sum_per_kernel = tf.reduce_sum(w_zeroed, axis=[0, 1, 2], keepdims=True)

        # Normalize each kernel to sum to 1
        return w_zeroed / (tf.abs(sum_per_kernel) + 1e-6)


class KernelDiversityLoss(keras.regularizers.Regularizer):

    def __init__(self, loss_constant_alpha, loss_constant_lambda, num_kernels, kernel_height, kernel_width):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.loss_constant_lambda = loss_constant_lambda
        self.loss_constant_alpha = loss_constant_alpha


    def __call__(self, kernel_tensor):
            # shape: (1, kernel_height, kernel_width, num_kernels)

            # shape (num_kernels, kernel_height * kernel_width)
            flattened_kernels = tf.transpose(tf.squeeze(tf.reshape(kernel_tensor, [1, self.kernel_height * self.kernel_width, self.num_kernels])))

            # Add small regularization to avoid singular matrices
            eps = 1e-8
            regularized_kernels = flattened_kernels + eps * tf.eye(tf.shape(flattened_kernels)[0], tf.shape(flattened_kernels)[1])

            try:
                singular_values = tf.linalg.svd(regularized_kernels, compute_uv=False)
                # Clamp singular values to avoid log(0)
                singular_values = tf.maximum(singular_values, eps)
                return - self.loss_constant_lambda * tf.reduce_sum(math.log(singular_values + self.loss_constant_alpha))
            except tf.errors.InvalidArgumentError:
                # Fallback: return a small penalty if SVD fails
                return tf.constant(0.1, dtype=tf.float32)
