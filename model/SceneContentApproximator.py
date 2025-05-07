import tensorflow as tf
from tensorflow import math
import keras
import numpy as np
from keras import Model, layers


class SceneContentApproximator(Model):



    def __init__(self, num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda):
        super(SceneContentApproximator, self).__init__()


        self.learning_rate = learning_rate
        self.conv = layers.Conv2D(num_kernels, (kernel_height,kernel_width), use_bias= False, kernel_regularizer = KernelDiversityLoss(loss_constant_alpha, loss_constant_lambda), kernel_constraint = KernelConstraint(), padding='same')

    
    def call(self, input):
        return self.conv(input)
    
    
    
    def train(self, image_batch, epochs):

        def custom_loss(y_true, y_pred):
            # y_true.shape (1, resize_height, resize_width, 1)
            # y_true.shape (1, resize_height, resize_width, num_kernels)

            # y_true gets broadcasted
            discrepancy_loss =  math.reduce_sum(math.square(y_true - y_pred))
            return discrepancy_loss
                                                



        optimizer = keras.optimizers.AdamW(self.learning_rate)

        @tf.function
        def train_step(image_batch):
            with tf.GradientTape() as tape:
                prediction = self(image_batch, training=True)
                loss = custom_loss(image_batch, prediction)
            
            # Get gradients and update weights
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for epoch in range (epochs):
            train_step(image_batch)



    def train_and_get(self, image_batch, epochs):
        self.train(image_batch, epochs)
        return self.call(image_batch)
    


class KernelConstraint(keras.constraints.Constraint):

    def __init__(self, ):
        super().__init__()


    def __call__(self, w):
            
        # shape of w: (kernel_height, kernel_width, num_colorchannels (1 in case of grayscaled), num_kernels)

        w_shape = w.shape
        middle_zero_tensor = np.ones(w_shape)

        middle_zero_tensor[int(w_shape[0]/2), int(w_shape[1]/2), :, :] = 0

        middle_zero_w =  w * middle_zero_tensor


        sums = tf.reduce_sum(middle_zero_w, [0,1,2], keepdims = True)   
        return middle_zero_w / sums

    
class KernelDiversityLoss(keras.regularizers.Regularizer):

    def __init__(self, loss_constant_alpha, loss_constant_lambda, num_kernels, kernel_height, kernel_width):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.loss_constant_lambda = loss_constant_lambda
        self.loss_constant_alpha = loss_constant_alpha


    def __call__(self, kernel_tensor):
            # shape: (1, kernel_height, kernel_width, num_kernels)

            # shape (kernel_height * kernel_width, num_kernels)
            flattened_kernels = tf.squeeze(tf.reshape(kernel_tensor, [1, self.kernel_height * self.kernel_width, self.num_kernels])) 



            singular_values = tf.linalg.svd(flattened_kernels)
            return - self.loss_constant_lambda * tf.reduce_sum(math.log(singular_values + self.loss_constant_alpha))