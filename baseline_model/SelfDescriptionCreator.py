import math
from keras import Model, layers
import keras
import tensorflow as tf
import numpy as np


class SelfDescriptionCreator(Model):
    def __init__(self, kernel_height, kernel_width, L, learning_rate):
        super(SelfDescriptionCreator, self).__init__()
        
        self.L = L
        self.learning_rate = learning_rate
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
            
            # Bilinear downsampling
            downsampled = tf.image.resize(
                inputs,
                target_size,
                method='bilinear',
                antialias=True
            )
            
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
    def train(self, image_batch, epochs):
                                                
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
        self.reset_conv_weights()
        residual_list = tf.unstack(image_batch)
        residual_list = [tf.expand_dims(x, axis=0) for x in residual_list]
        selfdescriptions_list = []
        i = 0
        for image in residual_list:
            self.train(image, epochs)
            weights = self.get_weights()[0]
            flattened_weights = tf.reshape(weights, [-1])
            selfdescriptions_list.append(flattened_weights)

        return tf.stack(selfdescriptions_list, axis=0)
    
    def reset_conv_weights(self):
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                layer.depthwise_kernel.assign(
                    layer.depthwise_initializer(tf.shape(layer.depthwise_kernel))
                )
                if layer.use_bias:
                    layer.bias.assign(
                        layer.bias_initializer(tf.shape(layer.bias))
                    )



class KernelConstraint(keras.constraints.Constraint):

    def __init__(self, kernel_shape):
        super().__init__()


        kernel_width, kernel_height, num_kernels, _ = kernel_shape
        mask = np.ones(kernel_shape)
        mask[kernel_height//2, kernel_width//2, :, :] = 0
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)


    def __call__(self, w):
            
        w_zeroed = w * self.mask

        # Compute sum of each kernel
        sum_per_kernel = tf.reduce_sum(w_zeroed, axis=[0, 1, 2], keepdims=True)

        # Normalize each kernel to sum to 1
        return w_zeroed / sum_per_kernel

    