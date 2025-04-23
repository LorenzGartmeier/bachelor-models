import tensorflow as tf
from matplotlib import pyplot as plt
import os
import keras
import numpy as np
from keras import Model, layers

resize_height, resize_width = 256, 256

def getDatasetFromDirectory(path, batch_size) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        directory=path,
        labels=None,
        color_mode="grayscale",
        image_size=(resize_height, resize_width),
        batch_size=batch_size
    )

dataset = getDatasetFromDirectory(os.path.join('datasets', 'coco2017'), 32)
print(dataset.shape)

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

#class ResidualCreator(Model):




