import tensorflow as tf
import os
AUTOTUNE = tf.data.AUTOTUNE


def load_image(path):
    img  = tf.io.read_file(path)
    img  = tf.image.decode_image(img, channels=1, expand_animations=False)
    img  = tf.image.convert_image_dtype(img, tf.float32)        # [0,1]
    return img              # shape=(H,W,1), ragged across dataset


def random_patch(img, patch_hw=256):
    h, w = tf.shape(img)[0], tf.shape(img)[1]

    # pad to at least 256×256 so tf.image.random_crop is always legal
    img = tf.image.resize_with_crop_or_pad(img,
                                           tf.maximum(h, patch_hw),
                                           tf.maximum(w, patch_hw))

    patch = tf.image.random_crop(img, size=[patch_hw, patch_hw, 1])
    return patch            # shape=(256,256,1)

def getDatasetFromDirectory(path, batch_size, patch_hw=256, repeat=True):
        # Define file patterns and list files
    patterns = [
        os.path.join(path, "*.jpg"),
        os.path.join(path, "*.jpeg"),
        os.path.join(path, "*.JPEG"),
        os.path.join(path, "*.png")
    ]
    files = tf.data.Dataset.list_files(patterns, shuffle=False)
    dataset    = files.map(load_image, num_parallel_calls=AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    dataset = (dataset
          .map(random_patch, num_parallel_calls=AUTOTUNE)
          .shuffle(10 * batch_size, reshuffle_each_iteration=True)
          .batch(batch_size, drop_remainder=True)
          .prefetch(AUTOTUNE))
    return dataset

