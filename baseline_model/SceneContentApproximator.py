import tensorflow as tf
from tensorflow import math
import keras
import numpy as np
from keras import Model, layers


class SceneContentApproximator(Model):



    def __init__(self, num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda, **kwargs):
        super(SceneContentApproximator, self).__init__(**kwargs)


        self.learning_rate = learning_rate
        self.conv = layers.Conv2D(
                                    num_kernels, 
                                    (kernel_height,kernel_width),
                                    use_bias= False, 
                                    kernel_regularizer = KernelDiversityLoss(loss_constant_alpha, loss_constant_lambda, num_kernels, kernel_height, kernel_width),
                                    kernel_constraint = KernelConstraint(), 
                                    padding='same' # to let the ouput of the layer have the same size as the input
                                    )
        
    def call(self, input):
        return self.conv(input)
    
    # expects grayscaled images (num_colorchannels = 1)
    def train(self, dataset, epochs):

        def custom_loss(y_true, y_pred):
            # y_true.shape (batch_size, resize_height, resize_width, num_colorchannels)
            # y_true.shape (batch_size, resize_height, resize_width, num_kernels)

            # y_true gets broadcasted (num_colorchannels = 1 required)
            discrepancy_loss =  math.reduce_sum(math.square(y_true - y_pred))
            return discrepancy_loss
                                                
        optimizer = keras.optimizers.AdamW(self.learning_rate)

        @tf.function
        def train_step(image_batch):
            with tf.GradientTape() as tape:

                # shape: ()
                prediction = self(image_batch, training=True)

                # single value
                loss = custom_loss(image_batch, prediction) + self.losses[0]
            
            # Get gradients and update weights
            gradients = tape.gradient(loss, self.trainable_variables)
            # Check if any gradient is None
            if any(grad is None for grad in gradients):
                tf.print("Warning: One or more gradients are None.")
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for i in range (epochs):
            tf.print("Epoch: ", i)
            for image_batch in dataset:
                train_step(image_batch)
    


class KernelConstraint(keras.constraints.Constraint):
    def __init__(self):
        super().__init__()


    def __call__(self, w):
        # Get the shape components dynamically
        kernel_height, kernel_width, num_colorchannels, num_kernels = tf.unstack(tf.shape(w))
        height_mid = kernel_height // 2
        width_mid = kernel_width // 2

        # Create indices for the center position (h_mid, w_mid)
        indices = tf.stack([height_mid, width_mid])
        indices = tf.reshape(indices, [1, 2])  # Shape (1, 2)

        # Create a 2D tensor with 1 at the center position
        updates = tf.ones([1], dtype=w.dtype)
        shape = tf.stack([kernel_height, kernel_width])
        center_spatial = tf.scatter_nd(indices, updates, shape)

        # Reshape to (h, w, 1, 1) for broadcasting
        center_spatial = tf.reshape(center_spatial, [kernel_height, kernel_width, 1, 1])

        # Tile to match input and output channels dimensions
        center_mask = tf.tile(center_spatial, [1, 1, num_colorchannels, num_kernels])

        # Create mask (0 at center, 1 elsewhere)
        mask = 1 - center_mask

        # Zero out the center positions in the kernel
        w_zeroed = w * mask

        # Compute sum of each kernel
        sum_per_kernel = tf.reduce_sum(w_zeroed, axis=[0, 1, 2], keepdims=True)

        # Normalize each kernel to sum to 1
        return w_zeroed / sum_per_kernel

    
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



            singular_values = tf.linalg.svd(flattened_kernels, compute_uv=False)
            return - self.loss_constant_lambda * tf.reduce_sum(math.log(singular_values + self.loss_constant_alpha))