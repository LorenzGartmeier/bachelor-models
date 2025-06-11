import tensorflow as tf
import glob
import os

def imagepathToTensor(imagepath) -> tf.Tensor:
    img = tf.io.read_file(imagepath)
    img = tf.io.decode_image(img, channels=1)  # Force grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img


def getDatasetFromDirectory(path, batch_size) -> tf.data.Dataset:
        # Define file patterns and list files
    patterns = [
        os.path.join(path, "*.jpg"),
        os.path.join(path, "*.jpeg"),
        os.path.join(path, "*.JPEG"),
        os.path.join(path, "*.png")
    ]
    file_ds = tf.data.Dataset.list_files(patterns, shuffle=False)
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