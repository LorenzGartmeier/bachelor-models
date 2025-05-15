import math
from keras import Model, layers
import keras
import tensorflow as tf
import numpy as np


class SelfDescriptionCreator(Model):
    def __init__(self, kernel_height, kernel_width):
        super(SelfDescriptionCreator, self).__init__()
        
        
        # Depthwise convolution (one 11x11 filter per residual channel)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_height,  kernel_width),
            depth_multiplier=1,
            padding='same',
            use_bias=False,
            activation=None
        )

    def call(self, inputs):
        # Input shape: (1, image_height, image_width, num_kernels)
        batch_size, image_height, image_width, num_kernels = tf.unstack(tf.shape(inputs))
        total_error = tf.zeros((batch_size, image_height, image_width), dtype=inputs.dtype)

        # Process each scale
        for l in range(1, self.L + 1):
            factor = 2 ** (l - 1)
            target_size = (image_height // factor, image_width // factor)
            
            # 1. Bilinear downsampling
            downsampled = tf.image.resize(
                inputs,
                target_size,
                method='bilinear',
                antialias=True
            )
            
            # 2. Apply 11x11 convolution
            r_hat = self.depthwise_conv(downsampled)
            
            # 3. Compute and upsample error
            error = downsampled - r_hat
            upsampled_error = tf.image.resize(
                error, 
                (image_height, image_width),
                method='bilinear'
            )
            
            # 4. Accumulate errors across residuals
            total_error += tf.reduce_sum(upsampled_error, axis=-1)

        # 5. Compute final loss (mean squared error)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(total_error), axis=[1, 2]))
        self.add_loss(loss)
        
        return inputs  # Maintain Keras functional API

    
    # expects a batch with shape (1, image_height, image_width, num_kernels)
    def train(self, image_batch, epochs, L):
                                                
        optimizer = keras.optimizers.AdamW(self.learning_rate)

        @tf.function
        def train_step(image_batch):
            with tf.GradientTape() as tape:

                # ignore prediction
                _ = self(image_batch, training=True)

                # single value
                loss = tf.add_n(self.losses)
            
            # Get gradients and update weights
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for _ in range (epochs):
            train_step(image_batch)

    # expects batches from the SceneContentApproximator of shape (batch_size, image_height, image_width, num_kernels)
    # returns a list of shape (batch_size, num_weights)
    def train_and_get(self, image_batch, epochs):
        residual_list = tf.unstack(image_batch, dim = 0)
        residual_list = [tf.expand_dims(x, axis=0) for x in list]
        selfdescriptions_list = []

        for image in residual_list:
            self.train(image, epochs)
            weights = self.get_weights()[0]
            flattened_weights = tf.reshape(weights, [-1])
            selfdescriptions_list.append(flattened_weights)

        return tf.stack(selfdescriptions_list, axis=0)
    


class KernelConstraint(keras.constraints.Constraint):

    def __init__(self, ):
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
        return w * mask

    