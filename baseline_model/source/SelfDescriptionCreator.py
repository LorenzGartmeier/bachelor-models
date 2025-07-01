import math
from keras import Model, layers
import keras
import tensorflow as tf
import numpy as np


class SelfDescriptionCreator(Model):
    def __init__(self, kernel_height, kernel_width, num_kernels, learning_rate):
        super(SelfDescriptionCreator, self).__init__()
        
        # num kernels is the number of residuals, determined by the prevous SceneContentApproximator
        self.num_kernels = num_kernels
        self.learning_rate = learning_rate

        self.kernel_height = kernel_height
        self.kernel_width = kernel_width

        self.optimizer = keras.optimizers.AdamW(learning_rate, weight_decay=0.003)

        self.kernel_constraint = KernelConstraint((kernel_height, kernel_width, num_kernels, 1)) 
        # Depthwise convolution (one 11x11 filter per residual channel)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_height,  kernel_width),
            depth_multiplier=1,
            padding='same',
            use_bias=False,
            activation=None,
            depthwise_constraint=self.kernel_constraint,
        )

    @tf.function(jit_compile=True)
    def _forward(self, residual):
        e = residual - self.depthwise_conv(residual)
        return tf.reduce_sum(e, axis=-1)

    @tf.function(jit_compile=True)
    def train_step(self, x):
        with tf.GradientTape() as tape:
            err  = self._forward(x)
            loss = tf.reduce_mean(tf.square(err))    
        g = tape.gradient(loss, self.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 10.0)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))

        return loss
    
    # expects a batch with shape (1, image_height, image_width, num_kernels)
    def train(self, residual, epochs):

        best_loss = tf.math.inf
        decay_patience = 150
        decay_wait = 0
        stop_patience = 300
        stop_wait = 0

        self.optimizer.learning_rate.assign(self.learning_rate)
        for _ in tf.range(epochs):
            loss = self.train_step(residual)
            if loss < best_loss:
                best_loss = loss
                decay_wait = 0
                stop_wait = 0
            
            else:
                decay_wait += 1
                stop_wait += 1
                if decay_wait >= decay_patience:
                    self.optimizer.learning_rate.assign(self.optimizer.learning_rate * 0.5)
                    decay_wait = 0
                if stop_wait >= stop_patience:
                    break


    # expects batches from the SceneContentApproximator of shape (1, image_height, image_width, num_kernels)
    # returns a tensor of shape (batch_size, num_weights) and a tensor of shape (batch_size,1) for the losses
    def train_and_get(self, residual, epochs):

        
        self.reset_conv_weights()
        self.train(residual, epochs)
        weights = self.depthwise_conv.kernel
        flattened_weights = tf.reshape(weights, [-1])
        return flattened_weights
    
    def reset_conv_weights(self):
        for v in self.depthwise_conv.weights:
            v.assign(tf.random.normal(v.shape, mean = 1 / (self.kernel_height * self.kernel_width), stddev=0.01))



class KernelConstraint(keras.constraints.Constraint):

    def __init__(self, kernel_shape):
        super().__init__()


        kernel_height, kernel_width, num_kernels, _ = kernel_shape
        mask = np.ones(kernel_shape)
        mask[kernel_height//2, kernel_width//2, :, :] = 0
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)


    def __call__(self, w):
            
        return w * self.mask


    