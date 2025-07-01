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
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
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
    @tf.function(jit_compile=True)
    def train(self, residual, epochs):

        best_loss = tf.constant(np.float32.max)
        decay_wait = tf.constant(0)
        stop_wait = tf.constant(0)
        decay_patience = tf.constant(150)
        stop_patience  = tf.constant(300)

        def cond(epoch, best_loss, decay_wait, stop_wait):
            return tf.logical_and(epoch < epochs,
                                  stop_wait < stop_patience)

        def body(epoch, best_loss, decay_wait, stop_wait):
            loss = self.train_step(residual)
            improved = loss < best_loss

            best_loss   = tf.where(improved, loss, best_loss)
            decay_wait  = tf.where(improved, 0, decay_wait + 1)
            stop_wait   = tf.where(improved, 0, stop_wait  + 1)

            def do_decay():
                self.optimizer.learning_rate.assign(self.optimizer.learning_rate * 0.5)
                return tf.constant(0)
            decay_wait = tf.cond(decay_wait >= decay_patience,
                                 do_decay,
                                 lambda: decay_wait)

            return epoch + 1, best_loss, decay_wait, stop_wait

        tf.while_loop(cond, body,
                      loop_vars=[0, best_loss, decay_wait, stop_wait])


    # expects batches from the SceneContentApproximator of shape (1, image_height, image_width, num_kernels)
    # returns a tensor of shape (batch_size, num_weights) and a tensor of shape (batch_size,1) for the losses
    @tf.function(jit_compile=True)
    def train_and_get(self, residual, epochs):

        
        self.reset_conv_weights()

        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

        self.optimizer.learning_rate.assign(self.learning_rate)
        self.train(residual, epochs)
        weights = self.depthwise_conv.kernel
        flattened_weights = tf.reshape(weights, [-1])
        return flattened_weights
    


    @tf.function(jit_compile=True)
    def reset_conv_weights(self):
        shape = tf.shape(self.depthwise_conv.kernel)
        init = tf.random.normal(shape,
                                mean=1.0/(self.kernel_height*self.kernel_width),
                                stddev=0.01)
        self.depthwise_conv.kernel.assign(init)



class KernelConstraint(keras.constraints.Constraint):

    def __init__(self, kernel_shape):
        super().__init__()


        kernel_height, kernel_width, num_kernels, _ = kernel_shape
        mask = np.ones(kernel_shape)
        mask[kernel_height//2, kernel_width//2, :, :] = 0
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)


    def __call__(self, w):
            
        return w * self.mask


    