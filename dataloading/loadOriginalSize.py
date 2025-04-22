import tensorflow as tf
import tensorflow_datasets as tfds

coco_dataset, coco_info = tfds.load(
    'coco/2017',
    with_info=True,
    as_supervised=False,  # This returns dictionaries rather than (image, label) tuples
)

print(coco_dataset)
