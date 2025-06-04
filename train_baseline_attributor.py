# Expects a trained and saved SceneContentApproximator in baseline_model/scene_content_approximator.keras

import tensorflow as tf
from baseline_model.Attributor import Attributor
from baseline_model.SceneContentApproximator import SceneContentApproximator
from baseline_model.SelfDescriptionCreator import SelfDescriptionCreator
from keras import datasets
from dataloading.loadResized import getDatasetFromDirectory
import os

from baseline_model.SceneContentApproximator import SceneContentApproximator
import tensorflow as tf

sceneContentApproximator = tf.keras.models.load_model(
    'baseline_model/scene_content_approximator.keras',
    custom_objects={'SceneContentApproximator': SceneContentApproximator}
)

spec = tf.TensorSpec(shape=(32, 121), dtype=tf.float32)


# Get the number of channels from the loaded model
num_kernels = sceneContentApproximator.conv.filters
selfDescriptionCreator = SelfDescriptionCreator(11, 11, num_kernels, 0.001)

# Calculate the expected selfdescription length
selfdescription_length = 11 * 11 * num_kernels
attributor = Attributor(10, selfdescription_length)

label_list = []
i = 0
for folder in os.listdir('datasets/OSMA/fake'):
    if os.path.isdir(os.path.join('datasets/OSMA/fake', folder)):
        print(folder)
        dataset = getDatasetFromDirectory(
            os.path.join('datasets/OSMA/fake', folder), 
            batch_size=32, 
            resize_height=255, 
            resize_width=255
        ).take(5)

        # Process each batch to create self-descriptions and create dataset on-the-fly
        def generate_selfdescriptions():
            for batch in dataset:
                residuals = batch - sceneContentApproximator(batch)
                selfdescriptions = selfDescriptionCreator.train_and_get(tf.squeeze(residuals), epochs=10)
                for description in selfdescriptions:
                    yield description
        
        selfdescriptions_dataset = tf.data.Dataset.from_generator(
            generate_selfdescriptions,
            output_signature=tf.TensorSpec(shape=(selfdescription_length,), dtype=tf.float32)
        )
        
        label_list.append(folder)
        # Train the attributor with the images from this folder
        attributor.add_gmm(i, selfdescriptions_dataset, 1)
        i += 1

#attributor.save('baseline_model/attributor.keras')