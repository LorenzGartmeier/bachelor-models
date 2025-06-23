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


    def call(self, inputs):
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
        self.add_loss(kl_loss)
        
        # Perform depthwise convolution
        return tf.nn.depthwise_conv2d(
            input=inputs,
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
                antialias=False
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
        loss = tf.reduce_mean(tf.reduce_mean(tf.square(total_error), axis=[1, 2]))
        self.add_loss(loss)
        
        return inputs  # Maintain Keras functional API

    


    @tf.function(reduce_retracing=True, jit_compile=True)
    def train_step(self, residual_batch):
        with tf.GradientTape() as tape:
            _ = self(residual_batch, training=True)          
            loss = tf.add_n(self.losses)          
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


    # expects a batch with shape (1, image_height, image_width, num_kernels)
    def train(self, image_batch, epochs):                                  
        for _ in range (epochs):
            self.train_step(image_batch)

    def train_and_get(self, image_batch, epochs):
        self.reset_conv_weights()
        residual_list = tf.unstack(image_batch)
        residual_list = [tf.expand_dims(x, axis=0) for x in residual_list]
        selfdescriptions_list = []
        for image in residual_list:
            self.train(image, epochs)
            locs = tf.reshape(self.depthwise_conv.posterior_loc, [-1])
            scales = tf.reshape(tf.nn.softplus(self.depthwise_conv.posterior_scale), [-1])
            sefldescription = tf.concat([locs, scales], axis=0)
            selfdescriptions_list.append(sefldescription)

        return tf.stack(selfdescriptions_list)

    def reset_conv_weights(self):
        self.depthwise_conv.posterior_loc.assign(tf.random.normal(self.depthwise_conv.posterior_loc.shape))
        self.depthwise_conv.posterior_scale.assign(tf.constant(-5.0, shape=self.depthwise_conv.posterior_scale.shape))