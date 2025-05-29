from keras import Model, layers

class SceneContentApproximator(Model):

    def __init__(self, num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda):
        super(SceneContentApproximator, self).__init__()


        self.learning_rate = learning_rate
        self.conv = layers.Conv2D(
                                    num_kernels, 
                                    (kernel_height,kernel_width),
                                    use_bias= False, 
                                    padding='same' # to let the ouput of the layer have the same size as the input
                                    )

        
        
        