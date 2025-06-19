import math
from keras import Model, layers
import keras
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

class BayesianDepthwiseConv2D(layers.Layer):
    def __init__(self, kernel_height, kernel_width, **kwargs):
        super().__init__(**kwargs)
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.padding = kwargs.get('padding', 'same')
        
    def build(self, input_shape):
        # Input channels (last dimension of input)
        num_kernels = input_shape[-1]
        
        # Kernel shape: [H, W, num_kernels, 1] for depthwise
        # num_kernels = num kernels of previous SceneContentApproximator
        kernel_shape = (self.kernel_height, self.kernel_width, num_kernels, 1)

        
        # Number of non-center elements per kernel
        self.non_center_elements = self.kernel_height * self.kernel_width - 1
        
        # Prior distribution (standard normal)
        self.prior = tfd.Normal(loc=0., scale=1.)
        
        # Posterior parameters (trainable)
        self.posterior_loc = self.add_weight(
            name='posterior_loc',
            shape=kernel_shape,
            initializer='random_normal',
            trainable=True)
        
        self.posterior_scale = self.add_weight(
            name='posterior_scale',
            shape=kernel_shape,
            initializer=tf.initializers.constant(-5.),  # Initialized for softplus(scale) ≈ 0.01
            trainable=True)
        
    
    def call(self, inputs):
        # Get posterior distribution
        posterior = tfd.Normal(loc=self.posterior_loc, 
                               scale=tf.nn.softplus(self.posterior_scale))
        

        v = posterior.sample()

                # Reconstruct full kernels
        kernel_shape = [self.kernel_height, self.kernel_width, 
                        inputs.shape[-1], 1]
        
        # Start with all zeros (including center)
        kernel = tf.zeros(kernel_shape, dtype=v.dtype)
        
        # Create indices for non-center positions
        h, w = self.kernel_size
        center_i, center_j = h//2, w//2
        indices = []
        for i in range(h):
            for j in range(w):
                if (i, j) != (center_i, center_j):
                    indices.append([i, j])
        indices = tf.constant(indices)
        
        # Scatter non-center values
        kernel = tf.tensor_scatter_nd_update(
            kernel,
            indices,
            v
        )
        
        # Normalize each kernel to sum to 1
        sums = tf.reduce_sum(kernel, axis=[0, 1], keepdims=True)
        # small constatnt to avoid division by zero
        kernel_normalized = kernel / (sums + 1e-7)

        kl_loss = tf.reduce_sum(tfd.kl_divergence(posterior, self.prior))
        self.add_loss(kl_loss)
        
        # Perform depthwise convolution
        return tf.nn.depthwise_conv2d(
            input=inputs,
            filter=kernel_normalized,
            strides=(1, 1, 1, 1),
            padding=self.padding.upper()
        )

class SelfDescriptionCreator(Model):
    def __init__(self, kernel_height, kernel_width, L, learning_rate):
        super(SelfDescriptionCreator, self).__init__()
        self.L = L
        self.learning_rate = learning_rate
        
        # Replace with Bayesian convolution
        self.depthwise_conv = BayesianDepthwiseConv2D(
            kernel_height, kernel_width,
            padding='same'
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

        return tf.stack(selfdescriptions_list)

    def reset_conv_weights(self):
        # Reset Bayesian convolution parameters
        if hasattr(self.depthwise_conv, 'kernel_mean'):
            kernel_shape = self.depthwise_conv.posterior_loc.shape
            self.depthwise_conv.kernel_mean.assign(
                tf.keras.initializers.GlorotUniform()(kernel_shape)
            )
            self.depthwise_conv.posterior_scale.assign(
                tf.constant_initializer(-10.0)(kernel_shape)  # Reset to low variance
        )