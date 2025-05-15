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
        # Input shape: (batch_size, image_height, image_width, num_kernels)
        batch_size, H, W, _ = tf.unstack(tf.shape(inputs))
        total_error = tf.zeros((batch_size, H, W), dtype=inputs.dtype)

        # Process each scale
        for l in range(1, self.L + 1):
            factor = 2 ** (l - 1)
            target_size = (H // factor, W // factor)
            
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
                (H, W),
                method='bilinear'
            )
            
            # 4. Accumulate errors across residuals
            total_error += tf.reduce_sum(upsampled_error, axis=-1)

        # 5. Compute final loss (mean squared error)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(total_error), axis=[1, 2]))
        self.add_loss(loss)
        
        return inputs  # Maintain Keras functional API

    
    # expects a batch with shape (batch_size, image_height, image_width, num_kernels)
    def train(self, image_batch, epochs, L):
                                                
        optimizer = keras.optimizers.AdamW(self.learning_rate)

        @tf.function
        def train_step(image_batch):
            with tf.GradientTape() as tape:

                # ignore prediction
                prediction = self(image_batch, training=True)

                # single value
                loss = self.losses[0]
            
            # Get gradients and update weights
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        

        image_height = len(image_batch[0])
        image_width = len(image_batch[0][0])
        for _ in range (epochs):
            for l in L:
                # Perform bilinear downsampling
                downsampled_batch = tf.image.resize(
                    image_batch,
                    size=(image_height*(2^(l-1)), image_width*(2^(l-1))),
                    method=tf.image.ResizeMethod.BILINEAR
                )
                train_step(downsampled_batch)

    # expects batches from the SceneContentApproximator of shape (batch_size, image_height, image_width, num_kernels)
    def train_and_get(self, image_batch, epochs):#
        self.train(image_batch, epochs)
        return self.get_weights()
    


class KernelConstraint(keras.constraints.Constraint):

    def __init__(self, ):
        super().__init__()


    def __call__(self, w):
            
        # shape of w: (kernel_height, kernel_width, num_colorchannels (1 in case of grayscaled), num_kernels)

        w_shape = w.shape
        middle_zero_tensor = np.ones(w_shape)

        middle_zero_tensor[int(w_shape[0]/2), int(w_shape[1]/2), :, :] = 0

        return w * middle_zero_tensor

    