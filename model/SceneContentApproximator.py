import tensorflow as tf
from tensorflow import math
import keras
import numpy as np
from keras import Model, layers


class SceneContentApproximator(Model):



    def __init__(self, num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda):
        super(SceneContentApproximator, self).__init__()


        self.learning_rate = learning_rate
        self.conv = layers.Conv2D(
                                    num_kernels, 
                                    (kernel_height,kernel_width),
                                    use_bias= False, 
                                    kernel_regularizer = KernelDiversityLoss(loss_constant_alpha, loss_constant_lambda, num_kernels, kernel_height, kernel_width),
                                    kernel_constraint = KernelConstraint(), 
                                    padding='same' # to let the ouput of the layer have the same size as the input
                                    )
        
    def call(self, input):
        return self.conv(input)
    
    # expects grayscaled images (num_colorchannels = 1)
    def train(self, dataset, epochs):

        def custom_loss(y_true, y_pred):
            # y_true.shape (batch_size, resize_height, resize_width, num_colorchannels)
            # y_true.shape (batch_size, resize_height, resize_width, num_kernels)

            # y_true gets broadcasted (num_colorchannels = 1 required)
            discrepancy_loss =  math.reduce_sum(math.square(y_true - y_pred))
            return discrepancy_loss
                                                
        optimizer = keras.optimizers.AdamW(self.learning_rate)

        @tf.function
        def train_step(image_batch):
            with tf.GradientTape() as tape:

                # shape: ()
                prediction = self(image_batch, training=True)

                # single value
                loss = custom_loss(image_batch, prediction)
            
            # Get gradients and update weights
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for _ in range (epochs):
            for image_batch in dataset:
                train_step(image_batch)


    # expects batches of shape (batch_size, image_height, image_width, num_colorchannels)
    # with num_colorchannels = 1 for grayscaled images
    # output: prediction batch with shape (batch_size, image_height, image_width, num_kernels)
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
            # the paper uses (num_kernels, kernel_height * kernel_width), the difference is to be examined
            flattened_kernels = tf.squeeze(tf.reshape(kernel_tensor, [1, self.kernel_height * self.kernel_width, self.num_kernels])) 



            singular_values = tf.linalg.svd(flattened_kernels)
            return - self.loss_constant_lambda * tf.reduce_sum(math.log(singular_values + self.loss_constant_alpha))