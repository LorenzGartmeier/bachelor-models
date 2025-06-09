from keras import layers
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np

class ConstrainedBayesianConv2D(layers.Layer):
    def __init__(self, kernel_height, kernel_width, kernel_regularizer=None, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        
        # Create center mask
        self.mask = np.ones((kernel_h-eight, kernel_width), dtype=np.float32)
        self.mask[kernel_height//2, kernel_width//2] = 0
        self.mask = tf.constant(self.mask)
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        
        # Create mask for broadcasting
        self.bcast_mask = tf.reshape(self.mask, [self.kernel_height, self.kernel_width, 1, 1])
        
        # Number of non-center elements per kernel
        self.non_center_elements = self.kernel_height * self.kernel_width - 1
        
        # Prior distribution (standard normal)
        self.prior = tfd.Normal(loc=0., scale=1.)
        
        self.posterior_mean = self.add_weight(
            name='posterior_loc',
            shape=[self.non_center_elements, in_channels, self.filters],
            initializer='random_normal',
            trainable=True)
        
        self.posterior_var = self.add_weight(
            name='posterior_scale',
            shape=[self.non_center_elements, in_channels, self.filters],
            initializer=tf.initializers.constant(-5.),  # Initialized for softplus(scale) ≈ 0.01
            trainable=True)
        
    def call(self, inputs, training=True):
        # Get posterior distribution
        posterior = tfd.Normal(loc=self.posterior_mean, 
                               scale=tf.nn.softplus(self.posterior_var))
        

        v = posterior.sample()


        # Reconstruct full kernels
        kernel_shape = [self.kernel_size[0], self.kernel_size[1], 
                        inputs.shape[-1], self.filters]
        
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
        
        # Apply constraints
        kernel_zeroed = kernel * tf.tile(
            self.bcast_mask, 
            [1, 1, kernel.shape[2], kernel.shape[3]]
        )
        
        # Normalize each kernel to sum to 1
        sums = tf.reduce_sum(kernel_zeroed, axis=[0, 1], keepdims=True)
        # small constatnt to avoid division by zero
        kernel_normalized = kernel_zeroed / (sums + 1e-7)
        
        if training and self.kernel_regularizer is not None:
            reg_loss = self.kernel_regularizer(kernel_normalized)
            self.add_loss(reg_loss)
        
        if training:
            kl_loss = tf.reduce_sum(tfd.kl_divergence(posterior, self.prior))
            self.add_loss(kl_loss)
        
        # Perform convolution
        return tf.nn.conv2d(
            inputs,
            kernel_normalized,
            strides=[1, 1, 1, 1],
            padding= "same"
        )