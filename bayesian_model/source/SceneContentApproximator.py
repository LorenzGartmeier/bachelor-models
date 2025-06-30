import tensorflow as tf
import tensorflow_probability as tfp
from keras import Model, layers, regularizers, initializers
import numpy as np
tfd = tfp.distributions

class ConstrainedBayesianConv2D(layers.Layer):
    def __init__(self, num_kernels, kernel_height, kernel_width,
                 kernel_regularizer=None, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.num_kernels     = num_kernels
        self.kernel_height   = kernel_height
        self.kernel_width    = kernel_width
        self.padding         = padding
        self.kernel_regularizer = kernel_regularizer


    def get_config(self):
        config = super().get_config()
        config.update(
            num_kernels   = self.num_kernels,
            kernel_height = self.kernel_height,
            kernel_width  = self.kernel_width,
            padding       = self.padding,
            kernel_regularizer = regularizers.serialize(self.kernel_regularizer)
        )
        return config

    @classmethod
    def from_config(cls, cfg):
        cfg['kernel_regularizer'] = regularizers.deserialize(
            cfg['kernel_regularizer'])
        return cls(**cfg)
        
    def build(self, input_shape):
        super().build(input_shape)

        # Number of non-center elements per kernel
        self.non_center_elements = self.kernel_height * self.kernel_width - 1
        
        # Prior distribution (standard normal)
        self.prior = tfd.Normal(loc=0., scale=1.)
        
        # Posterior parameters (trainable)
        self.posterior_loc = self.add_weight(
            name='posterior_loc',
            shape=[self.non_center_elements, 1, self.num_kernels],
            initializer='random_normal',
            trainable=True)
        
        self.posterior_scale = self.add_weight(
            name='posterior_scale',
            shape=[self.non_center_elements, 1, self.num_kernels],
            initializer=initializers.constant(-5.),  # Initialized for softplus(scale) ≈ 0.01
            trainable=True)
        
        self.kernel_shape = [self.kernel_height, self.kernel_width, 
                1, self.num_kernels]

        mask = np.ones(self.kernel_shape,
                       dtype=np.float32)
        center_h = self.kernel_height // 2
        center_w = self.kernel_width  // 2
        mask[center_h, center_w, :, :] = 0.0

        mask = tf.constant(mask)
        self.indices = tf.where(mask)
        
    @tf.function(
    input_signature=[tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32)]
    )
    def call(self, image_batch, training=None):
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

        sums = tf.reduce_sum(kernel, axis=[0, 1], keepdims=True)


        kernel_normalized = kernel / (sums + 1e-7)
        
        # Add regularization loss if in training mode
        if training and self.kernel_regularizer is not None:
            reg_loss = self.kernel_regularizer(kernel_normalized)
            self.add_loss(reg_loss)
        
        # Add KL divergence loss
        if training:
            kl_loss = tf.reduce_sum(tfd.kl_divergence(posterior, self.prior))
            self.add_loss(kl_loss)
        

        return tf.nn.conv2d(
            image_batch,
            kernel_normalized,
            strides=[1, 1, 1, 1],
            padding=self.padding.upper()
        )

@tf.keras.utils.register_keras_serializable()
class SceneContentApproximator(Model):
    def __init__(self, num_kernels, kernel_height, kernel_width,
                 learning_rate, loss_constant_alpha, loss_constant_lambda,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_kernels          = num_kernels
        self.kernel_height        = kernel_height
        self.kernel_width         = kernel_width
        self.learning_rate        = learning_rate
        self.loss_constant_alpha  = loss_constant_alpha
        self.loss_constant_lambda = loss_constant_lambda

        diversity_reg = KernelDiversityLoss(
            loss_constant_alpha=self.loss_constant_alpha,
            loss_constant_lambda=self.loss_constant_lambda,
            num_kernels=self.num_kernels,
            kernel_height=self.kernel_height,
            kernel_width=self.kernel_width
        )
        self.conv = ConstrainedBayesianConv2D(
            num_kernels=num_kernels,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            kernel_regularizer=diversity_reg,
            padding='same'
        )

        self.conv.build(input_shape=(None, None, None, self.num_kernels))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            num_kernels          = self.num_kernels,
            kernel_height        = self.kernel_height,
            kernel_width         = self.kernel_width,
            learning_rate        = self.learning_rate,
            loss_constant_alpha  = self.loss_constant_alpha,
            loss_constant_lambda = self.loss_constant_lambda,
        )
        return cfg
        

    @tf.function(
    input_signature=[tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32)]
    )
    def call(self, image_batch, training=None):
        return self.conv(image_batch, training=training)

    # Training method remains similar to original
    def train(self, dataset, epochs):
        optimizer = tf.keras.optimizers.AdamW(self.learning_rate)
        

        kl_factor = 1 / tf.data.experimental.cardinality(dataset).numpy()

        @tf.function(
        input_signature=[tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32)]
        )
        def train_step(image_batch):
            with tf.GradientTape() as tape:
                prediction = self(image_batch, training=True)
                
                # Reconstruction loss
                recon_loss = tf.reduce_sum(tf.square(image_batch - prediction))
                

                kernelDiversityLoss = self.losses[0]

                kl_loss = self.losses[1] * kl_factor 
                # Total loss = reconstruction + KL + regularization
                total_loss = recon_loss + kernelDiversityLoss + kl_loss

            gradients = tape.gradient(total_loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return recon_loss, kernelDiversityLoss, kl_loss, total_loss
        

        history = {
            'recon_loss': [],
            'kernel_diversity_loss': [],
            'kl_loss': [],
            'total_loss': []
        }

        for epoch in range(epochs):
            tf.print("Epoch: ", epoch)

            epoch_recon_loss = 0.0
            epoch_kernel_diversity_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_total_loss = 0.0
            for image_batch in dataset:
                recon_loss, kernelDiversityLoss, kl_loss, total_loss = train_step(image_batch)

                epoch_recon_loss += recon_loss
                epoch_kernel_diversity_loss += kernelDiversityLoss
                epoch_kl_loss += kl_loss
                epoch_total_loss += total_loss

            history['recon_loss'].append(epoch_recon_loss / kl_factor)
            history['kernel_diversity_loss'].append(epoch_kernel_diversity_loss / kl_factor)
            history['kl_loss'].append(epoch_kl_loss / kl_factor)
            history['total_loss'].append(epoch_total_loss / kl_factor)

        return history

    # returns a list of predictions, each with shape (batch_size, height, width, num_kernels)
    def predict(self, image_batch, n_samples=100):
        predictions = []
        
        for _ in range(n_samples):
            pred = self.conv(image_batch, training=True)
            predictions.append(pred)


        return predictions

@tf.keras.utils.register_keras_serializable()
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
            flattened_kernels = tf.reshape(kernel_tensor, [self.kernel_height * self.kernel_width, self.num_kernels])



            singular_values = tf.linalg.svd(flattened_kernels, compute_uv=False)
            return - self.loss_constant_lambda * tf.reduce_sum(tf.math.log(singular_values + self.loss_constant_alpha))

    def get_config(self):
        return {
            "loss_constant_lambda": self.loss_constant_lambda,
            "loss_constant_alpha":  self.loss_constant_alpha,
        }