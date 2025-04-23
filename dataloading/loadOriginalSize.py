import tensorflow as tf
import os

def imagepathToTensor(imagepath) -> tf.Tensor:
    img = tf.io.read_file(imagepath)
    img = tf.image.decode_image(img)
    img = tf.image.
    img = tf.image.rgb_to_grayscale(img)
    img = img /255

    return img


if __name__ == "__main__":
