import tensorflow as tf
import os

def imagepathToTensor(imagepath) -> tf.Tensor:
    img = tf.io.read_file(imagepath)
    img = tf.io.decode_image(img)
    img = tf.image.rgb_to_grayscale(img)

    # if normalization is desired
    # img = img /255

    return img


def getDatasetFromDirectory(path, batch_size) -> tf.data.Dataset:
    jpg_ds = tf.data.Dataset.list_files(os.path.join(path, "*/*.jpg"))
    png_ds = tf.data.Dataset.list_files(os.path.join(path, "*/*.png"))
    jpeg_ds = tf.data.Dataset.list_files(os.path.join(path, "*/*.jpeg"))

    file_ds = jpg_ds.concatenate(png_ds).concatenate(jpeg_ds)
    image_ds = file_ds.map(
    imagepathToTensor,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False
    )

    padded_ds =  image_ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=[None, None, 1],  
        padding_values=0.0
    )

    return padded_ds