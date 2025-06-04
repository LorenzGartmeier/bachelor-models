import tensorflow as tf
import tensorflow_probability as tfp
from keras import Model, layers, regularizers
import numpy as np
tfd = tfp.distributions

class ConstrainedBayesianConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, kernel_regularizer=None, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        
        # Create center mask
        h, w = kernel_size
        self.mask = np.ones((h, w), dtype=np.float32)
        self.mask[h//2, w//2] = 0
        self.mask = tf.constant(self.mask)
        
    def build(self, input_shape):
        h, w = self.kernel_size
        in_channels = input_shape[-1]
        
        # Create mask for broadcasting
        self.bcast_mask = tf.reshape(self.mask, [h, w, 1, 1])
        
        # Number of non-center elements per kernel
        self.non_center_elements = h * w - 1
        
        # Prior distribution (standard normal)
        self.prior = tfd.Normal(loc=0., scale=1.)
        
        # Posterior parameters (trainable)
        self.posterior_loc = self.add_weight(
            name='posterior_loc',
            shape=[self.non_center_elements, in_channels, self.filters],
            initializer='random_normal',
            trainable=True)
        
        self.posterior_scale = self.add_weight(
            name='posterior_scale',
            shape=[self.non_center_elements, in_channels, self.filters],
            initializer=tf.initializers.constant(-5.),  # Initialized for softplus(scale) ≈ 0.01
            trainable=True)
        
    def call(self, inputs, training=None):
        # Get posterior distribution
        posterior = tfd.Normal(loc=self.posterior_loc, 
                               scale=tf.nn.softplus(self.posterior_scale))
        

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
        
        # Add regularization loss if in training mode
        if training and self.kernel_regularizer is not None:
            reg_loss = self.kernel_regularizer(kernel_normalized)
            self.add_loss(reg_loss)
        
        # Add KL divergence loss
        if training:
            kl_loss = tf.reduce_sum(tfd.kl_divergence(posterior, self.prior))
            self.add_loss(kl_loss)
        
        # Perform convolution
        return tf.nn.conv2d(
            inputs,
            kernel_normalized,
            strides=[1, 1, 1, 1],
            padding=self.padding.upper()
        )

class SceneContentApproximator(Model):
    def __init__(self, num_kernels, kernel_height, kernel_width, 
                 learning_rate, loss_constant_alpha, loss_constant_lambda, **kwargs):
        super().__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.learning_rate = learning_rate
        self.loss_constant_alpha = loss_constant_alpha
        self.loss_constant_lambda = loss_constant_lambda
        
        # Create the regularizer
        diversity_reg = KernelDiversityLoss(
            loss_constant_alpha, loss_constant_lambda, 
            num_kernels, kernel_height, kernel_width
        )
        
        # Bayesian convolution layer
        self.conv = ConstrainedBayesianConv2D(
            filters=num_kernels,
            kernel_size=(kernel_height, kernel_width),
            kernel_regularizer=diversity_reg,
            padding='same'
        )
        
    def call(self, inputs):
        return self.conv(inputs)
    
    # Training method remains similar to original
    def train(self, dataset, epochs):
        optimizer = tf.keras.optimizers.AdamW(self.learning_rate)
        
        @tf.function
        def train_step(image_batch):
            with tf.GradientTape() as tape:
                # Forward pass
                prediction = self(image_batch, training=True)
                
                # Reconstruction loss
                recon_loss = tf.reduce_sum(tf.square(image_batch - prediction))
                
                # Total loss = reconstruction + KL + regularization
                total_loss = recon_loss + tf.reduce_sum(self.losses)
            
            # Get gradients and update weights
            gradients = tape.gradient(total_loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for epoch in range(epochs):
            tf.print("Epoch: ", epoch)
            for image_batch in dataset:
                train_step(image_batch)

    def predict_with_uncertainty(self, x, n_samples=100):
        predictions = []
        
        # Force training=True to enable sampling
        for _ in range(n_samples):
            pred = self.conv(x, training=True)
            predictions.append(pred)
        
        predictions = tf.stack(predictions)
        
        # Calculate statistics
        mean_pred = tf.reduce_mean(predictions, axis=0)
        predictive_variance = tf.reduce_mean(tf.square(predictions - mean_pred), axis=0)
        predictive_std = tf.sqrt(predictive_variance)
        
        return mean_pred, predictive_std

class KernelDiversityLoss(regularizers.Regularizer):

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
            return - self.loss_constant_lambda * tf.reduce_sum(tf.math.log(singular_values + self.loss_constant_alpha))
