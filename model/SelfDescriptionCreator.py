import math
from keras import Model, layers
import keras
import tensorflow as tf
import numpy as np


class SelfDescriptionCreator(Model):
    def __init__(self, B=11, L=3, K=8):
        super(SelfDescriptionCreator, self).__init__()
        self.B = B  # 11x11 neighborhood size
        self.L = L  # 3 scales
        self.K = K  # 8 residuals
        
        # Depthwise convolution (one 11x11 filter per residual channel)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(B, B),
            depth_multiplier=1,
            padding='same',
            use_bias=False,
            activation=None
        )

    def call(self, inputs):
        # Input shape: (1, image_height, image_width, num_kernels)
        batch_size, image_height, image_width, nunm_kernels = tf.unstack(tf.shape(inputs))
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
            
        # shape of w: (kernel_height, kernel_width, num_colorchannels (1 in case of grayscaled), num_kernels)

        w_shape = w.shape
        middle_zero_tensor = np.ones(w_shape)

        middle_zero_tensor[int(w_shape[0]/2), int(w_shape[1]/2), :, :] = 0

        return w * middle_zero_tensor

    