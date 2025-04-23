import tensorflow as tf
from tensorflow import math
from matplotlib import pyplot as plt
import os
import keras
import numpy as np
from keras import Model, layers


resize_height, resize_width = 256, 256

# should be odd numbers
kernel_height, kernel_width = 11, 11
learning_rate = 0.001

def getDatasetFromDirectory(path, batch_size) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        directory=path,
        labels= None,
        color_mode="grayscale",
        image_size=(resize_height, resize_width),
        batch_size=batch_size
    )

dataset = getDatasetFromDirectory(os.path.join('datasets', 'coco2017'), 32)

dataset = dataset.take(512)

iterator = dataset.as_numpy_iterator()
batch = iterator.next()
print(batch.shape) 
print(batch[0].shape)

fig, ax = plt.subplots(ncols=5, figsize=(20, 20))
for idx, img in enumerate(batch[:5]):
    img_display = np.squeeze(img)
    ax[idx].imshow(img_display, cmap='gray') 
    ax[idx].axis('off') 

plt.show()  



class KernelConstraint(keras.constraints.Constraint):

    def __call__(self, w):
        
        print("the shape of the kernel is ")
        print(tf.shape(w))
        w[int(kernel_height/2)][int(kernel_width/2)] = 0
        return w


class SceneContentApproximator(Model):

    def __init__(self):
        super(SceneContentApproximator, self).__init__()
        self.conv = layers.Conv2D(1, (kernel_height,kernel_width), use_bias= False, kernel_constraint = KernelConstraint())

    
    def call(self, input):
        return self.conv(input)
    
    


def trainResidualCreator(dataset, epochs = 10):
    residualCreator = SceneContentApproximator()


    def custom_loss(y_true, y_pred):
        # y_true: (resize_height, resize_width, 1)
        # y_pred: (resize_height - kernel_height + 1, resize_width - kernel_width + 1, 1)

        # cut off edges
        batch_size = len(y_true)
        y_true = y_true[:batch_size,int(kernel_height/2):-int(kernel_height/2),int(kernel_height/2):-int(kernel_height/2)]
        return math.reduce_sum(math.square(y_true - y_pred))


    optimizer = keras.optimizers.AdamW(learning_rate)

    @tf.function
    def train_step(images):
        with tf.GradientTape() as tape:
            predictions = residualCreator(images, training=True)
            loss = custom_loss(images, predictions)
        
        # Get gradients and update weights
        gradients = tape.gradient(loss, residualCreator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, residualCreator.trainable_variables))
    
    for epoch in range (epochs):
        print(epoch)
        for images in dataset:
            train_step(images)
    
    return residualCreator

residualCreator = trainResidualCreator(dataset,2)

batch = iterator.next()
sample_image = batch[0]
sample_residual = residualCreator(batch)[0]

fig, ax = plt.subplots(ncols=2, figsize=(20, 20))
ax[0].imshow(np.squeeze(sample_image), cmap = 'gray')
ax[0].axis('off')
ax[1].imshow(np.squeeze(sample_residual), cmap = 'gray')
ax[1].axis('off')

plt.show()  

print(residualCreator.get_weights())





        









