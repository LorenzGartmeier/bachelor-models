import math
from keras import Model, layers
import keras
import tensorflow as tf


class SelfDescriptionCreator(Model):
    def __init__(self, num_kernels, kernel_height, kernel_width, learning_rate):
        super(SelfDescriptionCreator, self).__init__()


        self.learning_rate = learning_rate
        self.conv = layers.Conv2D(
                                    num_kernels, 
                                    (kernel_height,kernel_width),
                                    use_bias= False, 
                                    kernel_constraint = KernelConstraint(), 
                                    padding='same' # to let the ouput of the layer have the same size as the input
                                    )
        
    def call(self, input):
        return self.conv(input)
    
    # expects a batch with shape (batch_size, image_height, image_width, num_kernels)
    def train(self, image_batch, epochs):

        def custom_loss(y_true, y_pred):
            # y_true.shape (batch_size, image_height, image_width, num_colorchannels)
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
        
        for epoch in range (epochs):
            train_step(image_batch)

    # expects batches from the SceneContentApproximator of shape (batch_size, image_height, image_width, num_kernels)
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

        return w * middle_zero_tensor

    