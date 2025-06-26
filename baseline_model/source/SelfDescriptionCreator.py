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
        loss_history = []

        for _ in tf.range(epochs):
            loss_history.append(self.train_step(residual))

        return loss_history

    # expects batches from the SceneContentApproximator of shape (batch_size, image_height, image_width, num_kernels)
    # returns a tensor of shape (batch_size, num_weights) and a tensor of shape (batch_size,1) for the losses
    def train_and_get(self, residual_batch, epochs):

        selfdescriptions_list = []
        loss_list = []
        
        for i in tf.range(tf.shape(residual_batch)[0]):
            self.reset_conv_weights()
            loss = self.train(residual_batch[i:i+1], epochs)
            weights = self.get_weights()[0]
            flattened_weights = tf.reshape(weights, [-1])
            selfdescriptions_list.append(flattened_weights)
            loss_list.append(loss)

        return tf.stack(selfdescriptions_list, axis=0), loss_list
    
    def reset_conv_weights(self):
        for v in self.depthwise_conv.weights:
            v.assign(tf.random.normal(v.shape, stddev=0.05))



class KernelConstraint(keras.constraints.Constraint):

    def __init__(self, kernel_shape):
        super().__init__()


        kernel_height, kernel_width, num_kernels, _ = kernel_shape
        mask = np.ones(kernel_shape)
        mask[kernel_height//2, kernel_width//2, :, :] = 0
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)


    def __call__(self, w):
            
        return w * self.mask


    