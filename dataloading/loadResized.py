import tensorflow as tf
import os
import keras
from keras import datasets

def getDatasetFromDirectory(path, batch_size) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(directory = path,
                                                       labels = None,
                                                       color_mode = "grayscale",
                                                       batch_size = batch_size
                                                       )
    