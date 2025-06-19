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
                                    kernel_initializer= keras.initializers.RandomNormal(stddev=0.05),
                                    kernel_regularizer = KernelDiversityLoss(loss_constant_alpha, loss_constant_lambda, num_kernels, kernel_height, kernel_width),
                                    kernel_constraint = KernelConstraint((kernel_height, kernel_width, 1, num_kernels)), # num_colorchannels = 1 for grayscaled images
                                    padding='same' # to let the ouput of the layer have the same size as the input
                                    )
        
    def call(self, input):
        return self.conv(input)
    
    # expects grayscaled images (num_colorchannels = 1)
    def train(self, dataset, epochs):

        num_batches = tf.data.experimental.cardinality(dataset).numpy()


        history = {
            'recon_loss': [],
            'kernel_diversity_loss': [],
            'total_loss': []
        }

        def custom_loss(y_true, y_pred):
            # y_true.shape (batch_size, resize_height, resize_width, num_colorchannels)
            # y_true.shape (batch_size, resize_height, resize_width, num_kernels)

            # y_true gets broadcasted (num_colorchannels = 1 required)
            return tf.reduce_mean(tf.square(y_true - y_pred))
                                                
        optimizer = keras.optimizers.AdamW(self.learning_rate, weight_decay=0.003) 

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
            gradients, _ = tf.clip_by_global_norm(gradients, 2.0)
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
            tf.print(f"Epoch {i+1}/{epochs} - Recon Loss: {epoch_recon_loss / num_batches:.4f}, "
                     f"Kernel Diversity Loss: {epoch_kernel_diversity_loss / num_batches:.4f}, "
                     f"Total Loss: {epoch_total_loss / num_batches:.4f}")

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


        w = w * self.mask

        # w = tf.clip_by_value(w, 0.0, tf.float32.max)  # clip to [0, inf) range

        # w = tf.clip_by_value(w, -1.0, 1.0)  # clip to [-1, 1] range
        sum_per_kernel = tf.reduce_sum(w, axis=[0, 1, 2], keepdims=True)
        eps = 1e-6                                   # or 1e-7, whatever you like
        # sum_per_kernel has shape (1, 1, 1, K)
        sign  = tf.where(sum_per_kernel >= 0, 1.0, -1.0)
        mag   = tf.maximum(tf.abs(sum_per_kernel), eps)   # at least eps


        safe_denominator = sign * mag                    # keeps the sign

        w = w / safe_denominator  
        
                    # normalise


        return w


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
            flattened_kernels = tf.squeeze(tf.reshape(kernel_tensor, [1, self.kernel_height * self.kernel_width, self.num_kernels]))

            singular_values = tf.linalg.svd(flattened_kernels, compute_uv=False)

            return - self.loss_constant_lambda * tf.reduce_sum(math.log(singular_values + self.loss_constant_alpha))
