import tensorflow as tf
import os
import keras
from keras import datasets

def getDatasetFromDirectory(path, batch_size, resize_height = 255, resize_width = 255) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        directory=path,
        labels= None,
        color_mode="grayscale",
        image_size=(resize_height, resize_width),
        batch_size=batch_size
    )

def getLabeledDatasetFromDirectory(path, batch_size, resize_height = 255, resize_width = 255) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        directory=path,
        labels= "inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=(resize_height, resize_width),
        batch_size=batch_size
    )