import tensorflow as tf
from tensorflow import math
from matplotlib import pyplot as plt
import os
import keras
import numpy as np
from keras import Model, layers
from dataloading.loadResized import getDatasetFromDirectory
from baseline_model.SceneContentApproximator import SceneContentApproximator

resize_height, resize_width = 256, 256

# should be odd numbers
kernel_height, kernel_width = 11, 11
# in the paper detoned as K
num_kernels = 4

learning_rate = 0.001

# please refer to the paper
loss_constant_alpha = 0.01

# please refer to the paper 
loss_constant_lambda = 1.0




dataset = getDatasetFromDirectory(os.path.join('datasets', 'train2017'), 32)

dataset = dataset.take(512)

# batches obtained with iterator.next() have shape (batch_size, image_height, image_width, num_colorchannels)
iterator = dataset.as_numpy_iterator()


fig, ax = plt.subplots(ncols=5, figsize=(20, 20))
for idx in range(5):
    img = iterator.next()[0]
    img_display = np.squeeze(img)
    ax[idx].imshow(img_display, cmap='gray') 
    ax[idx].axis('off') 

plt.show()  
        

batch = iterator.next()
image = np.squeeze(batch[0])

sceneContentApproximator = SceneContentApproximator(num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda)
sceneContentApproximator.train(dataset, 10)

prediction_batch = sceneContentApproximator(batch)


weights = sceneContentApproximator.get_weights()[0]
weights_max =tf.reduce_max(weights).numpy()
weights_min = tf.reduce_min(weights).numpy()


sums = tf.reduce_sum(weights, axis=[0, 1, 2])


fig, ax = plt.subplots(nrows = num_kernels, ncols = 4, figsize=(20, 20))
for i in range(num_kernels):
    kernel = np.squeeze(weights[:,:,:,i]) 
    # kernel /= max(abs(weights_min), abs(weights_max))
    approximated_image = np.squeeze(prediction_batch[0, :, :, i])
    residual = image - approximated_image
    ax[i, 0].imshow(kernel, cmap = 'bwr', vmin = weights_max - 0.001, vmax = weights_max)
    ax[i, 1].imshow(image, cmap = 'gray')
    ax[i, 2].imshow(approximated_image, cmap = 'gray')
    ax[i, 3].imshow(residual, cmap = 'gray')

plt.show()  

        









