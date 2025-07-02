import math
from keras import Model, layers
import keras
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

class BayesianDepthwiseConv2D(layers.Layer):
    def __init__(self, kernel_height, kernel_width, num_kernels, **kwargs):
        super().__init__(**kwargs)
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.num_kernels = num_kernels
        
    def build(self):
        
        # Number of non-center elements per kernel
        self.non_center_elements = self.kernel_height * self.kernel_width - 1
        self.kernel_shape = [self.kernel_height, self.kernel_width, 
                             self.num_kernels, 1]

        
        # Prior distribution (standard normal)
        self.prior = tfd.Normal(loc=0., scale=1.)
        
        # Posterior parameters (trainable)
        self.posterior_loc = self.add_weight(
            name='posterior_loc',
            shape=[self.non_center_elements * self.num_kernels],
            initializer='random_normal',
            trainable=True)
        
        self.posterior_scale = self.add_weight(
            name='posterior_scale',
            shape=[self.non_center_elements * self.num_kernels],
            initializer=tf.initializers.constant(-5.),  # Initialized for softplus(scale) ≈ 0.01
            trainable=True)

        mask = np.ones(self.kernel_shape, np.float32)
        mask[self.kernel_height//2, self.kernel_width//2, :, :] = 0.
        mask = tf.constant(mask)
        self.indices = tf.where(mask)


    def call(self, residual):
        # Get posterior distribution
        posterior = tfd.Normal(loc=self.posterior_loc, 
                               scale=tf.nn.softplus(self.posterior_scale))
        

        v = posterior.sample()
        
        # Start with all zeros (including center)
        kernel = tf.zeros(self.kernel_shape, dtype=v.dtype)
        
        
        
        # Scatter non-center values
        kernel = tf.tensor_scatter_nd_update(
            kernel,
            self.indices,
            v
        )
        
        # Normalize each kernel to sum to 1
        sums = tf.reduce_sum(kernel, axis=[0, 1], keepdims=True)
        # small constatnt to avoid division by zero
        kernel_normalized = kernel / (sums + 1e-7)

        kl_loss = tf.reduce_sum(tfd.kl_divergence(posterior, self.prior))
        kl_factor = tf.cast(tf.size(v), tf.float32)
        self.add_loss(kl_loss * kl_factor)
        
        # Perform depthwise convolution
        return tf.nn.depthwise_conv2d(
            input=residual,
            filter=kernel_normalized,
            strides=(1, 1, 1, 1),
            padding="SAME"
        )

class SelfDescriptionCreator(Model):
    def __init__(self, kernel_height, kernel_width, num_kernels, L, learning_rate):
        super(SelfDescriptionCreator, self).__init__()
        self.L = L
        self.learning_rate = learning_rate
        
        # Replace with Bayesian convolution
        self.depthwise_conv = BayesianDepthwiseConv2D(
            kernel_height, kernel_width, num_kernels
        )

        self.depthwise_conv.build()

        self.optimizer = keras.optimizers.AdamW(self.learning_rate)


    def call(self, residual):
            
        approximation = self.depthwise_conv(residual)
        loss = tf.reduce_sum(tf.square(tf.reduce_sum(residual - approximation, axis = -1)))
        self.add_loss(loss)
        return residual 

    


    @tf.function(jit_compile=True)
    def train_step(self, residual_batch):
        with tf.GradientTape() as tape:
            _ = self(residual_batch, training=True)          
            loss = tf.add_n(self.losses)          
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


    # expects a batch with shape (1, image_height, image_width, num_kernels)
    def train(self, image_batch, epochs):   
        decay_wait = 0
        decay_patience = 100
        stop_wait = 0
        stop_patience = 0
        best_loss = tf.float32.max
        self.optimizer.learning_rate.assign(self.learning_rate)                               
        for _ in range (epochs):
            loss = self.train_step(image_batch)

            if(loss < best_loss):
                best_loss = loss
                decay_wait = 0
                stop_wait = 0
            else: 
                decay_wait += 1
                stop_wait += 1

            if(decay_wait >= decay_patience):
                self.optimizer.learning_rate.assign(self.optimizer.learning_rate * 0.5)
            
            if(stop_wait >= stop_patience):
                break



    # a expects a residual of shape (1, height, width, num_kernels)
    def train_and_get(self, residual, epochs):
        self.reset_conv_weights()

        for var in self.optimizer.variables:
            var.assign(tf.zeros_like(var))
        self.train(residual, epochs)
        posterior = tfd.Normal(loc=self.posterior_loc, 
                               scale=tf.nn.softplus(self.posterior_scale))
        v = posterior.sample()
        return tf.reshape(v, [-1])


    def reset_conv_weights(self):
        self.depthwise_conv.posterior_loc.assign(tf.random.normal(self.depthwise_conv.posterior_loc.shape))
        self.depthwise_conv.posterior_scale.assign(tf.constant(-5.0, shape=self.depthwise_conv.posterior_scale.shape))