import math
from keras import Model, layers
import keras
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

class BayesianDepthwiseConv2D(layers.Layer):
    def __init__(self, kernel_size, prior_stddev=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.prior_stddev = prior_stddev
        self.padding = kwargs.get('padding', 'same')
        
    def build(self, input_shape):
        # Input channels (last dimension of input)
        in_channels = input_shape[-1]
        
        # Kernel shape: [H, W, in_channels, 1] for depthwise
        kernel_shape = (*self.kernel_size, in_channels, 1)
        
        # Create mask to zero out center weights (same as original constraint)
        self.mask = self.create_center_mask(kernel_shape)
        
        # Variational parameters (posterior)
        self.kernel_mean = self.add_weight(
            name='kernel_mean',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True
        )
        self.kernel_logvar = self.add_weight(
            name='kernel_logvar',
            shape=kernel_shape,
            initializer=tf.constant_initializer(-10.0),  # Small initial variance
            trainable=True
        )
        
        # Prior distribution (zero-mean Gaussian)
        self.prior = tfd.Independent(
            tfd.Normal(loc=0., scale=self.prior_stddev),
            reinterpreted_batch_ndims=4
        )
        
    def create_center_mask(self, kernel_shape):
        k_h, k_w, num_channels, _ = kernel_shape
        height_mid, width_mid = k_h // 2, k_w // 2
        
        # Create spatial mask (1 everywhere except center)
        spatial_mask = tf.ones((k_h, k_w), dtype=tf.float32)
        spatial_mask = tf.tensor_scatter_nd_update(
            spatial_mask,
            [[height_mid, width_mid]],
            [0.0]
        )
        
        # Expand dimensions for broadcasting
        return tf.reshape(spatial_mask, (k_h, k_w, 1, 1))
    
    def call(self, inputs):
        # Reparameterization trick
        kernel = self.kernel_mean + tf.exp(0.5 * self.kernel_logvar) * tf.random.normal(tf.shape(self.kernel_mean))
        
        # Apply center constraint
        kernel = kernel * self.mask
        
        # Compute KL divergence: KL(q(w) || p(w))
        posterior = tfd.Independent(
            tfd.Normal(loc=self.kernel_mean, scale=tf.exp(0.5 * self.kernel_logvar)),
            reinterpreted_batch_ndims=4
        )
        kl = tf.reduce_sum(tfd.kl_divergence(posterior, self.prior))
        self.add_loss(kl)
        
        # Perform depthwise convolution
        return tf.nn.depthwise_conv2d(
            input=inputs,
            filter=kernel,
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
            kernel_size=(kernel_height, kernel_width),
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
            kernel_shape = self.depthwise_conv.kernel_mean.shape
            self.depthwise_conv.kernel_mean.assign(
                tf.keras.initializers.GlorotUniform()(kernel_shape)
            )
            self.depthwise_conv.kernel_logvar.assign(
                tf.constant_initializer(-10.0)(kernel_shape)  # Reset to low variance
        )