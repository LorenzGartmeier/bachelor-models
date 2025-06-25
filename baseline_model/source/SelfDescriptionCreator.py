import math
from keras import Model, layers
import keras
import tensorflow as tf
import numpy as np


class SelfDescriptionCreator(Model):
    def __init__(self, kernel_height, kernel_width, num_kernels, L, learning_rate, beta):
        super(SelfDescriptionCreator, self).__init__()
        
        self.L = L
        self.beta = beta

        # num kernels is the number of residuals, determined by the prevous SceneContentApproximator
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.num_kernels = num_kernels
        self.learning_rate = learning_rate

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
    def _forward(self, resized_list, height, width):
        total   = tf.zeros([1, height, width], tf.float32)

        resized_list = resized_list.copy()
        def cond(idx, _):
            return tf.less(idx, self.L)          

        def body(idx, acc):
            r   = resized_list.pop()
            e   = r - self.depthwise_conv(r)
            acc = acc + tf.reduce_sum(
                        tf.image.resize(e, (height, width), "bilinear"), axis=-1)
            return idx + 1, acc

        _, total = tf.while_loop(
            cond, body,
            loop_vars=[tf.constant(0), total],
            maximum_iterations=self.L,         
            parallel_iterations=1)

        return total

    @tf.function(jit_compile=True)
    def train_step(self, x, height, width):
        with tf.GradientTape() as tape:
            err  = self._forward(x, height, width)
            loss = tf.reduce_mean(tf.square(err))      # MSE
            loss += self.beta * tf.add_n(
                        [tf.nn.l2_loss(v) for v in self.depthwise_conv.weights])
        g = tape.gradient(loss, self.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 10.0)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        return loss
    
    # expects a batch with shape (1, image_height, image_width, num_kernels)
    def train(self, image, height, width, epochs):
        resized_list = []
        l = tf.constant(1)
        while l < self.L + 1:
            f = 2 ** (l - 1)
            resized_image = tf.image.resize(image, (height // f, width // f), "bilinear")
            resized_list.append(resized_image)
            l += 1
        
        i = tf.constant(0)
        loss = tf.constant(0, dtype= tf.float32)
        while i < epochs:
            loss = self.train_step(resized_list, height, width)
            i += 1
        return loss

    # expects batches from the SceneContentApproximator of shape (batch_size, image_height, image_width, num_kernels)
    # returns a tensor of shape (batch_size, num_weights) and a tensor of shape (batch_size,1) for the losses
    def train_and_get(self, image_batch, epochs):
        residual_list = tf.unstack(image_batch)
        residual_list = [tf.expand_dims(x, axis=0) for x in residual_list]
        batch_size = len(residual_list)
        tensor_list = tf.TensorArray(dtype = image_batch.dtype, size = batch_size)
        for i, residual in enumerate(residual_list):
            tensor_list = tensor_list.write(i, residual)

        selfdescriptions_list = tf.TensorArray(dtype = tf.float32, size = batch_size, element_shape=tf.TensorShape([self.kernel_height * self.kernel_width * self.num_kernels]))
        loss_list = tf.TensorArray(dtype = tf.float32, size = batch_size)

        i = tf.constant(0)
        while i < batch_size:
            self.reset_conv_weights()
            image = tensor_list.read(i)
            height, width = tf.shape(image)[1], tf.shape(image)[2]
            loss = self.train(image, height, width, epochs)
            flattened_weights = tf.reshape(self.depthwise_conv.kernel, [-1])
            selfdescriptions_list = selfdescriptions_list.write(i, flattened_weights)
            loss_list = loss_list.write(i, loss)
            i += 1

        return selfdescriptions_list.stack(), loss_list.stack()
    
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
    



    