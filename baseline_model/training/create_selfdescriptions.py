# Expects a trained and saved SceneContentApproximator in baseline_model/scene_content_approximator.keras
import sys
import os

project_root = os.environ.get('BACHELOR_MODELS_ROOT', '.')
sys.path.append(project_root)

import tensorflow as tf
from baseline_model.source.SceneContentApproximator import SceneContentApproximator
from baseline_model.source.SelfDescriptionCreator import SelfDescriptionCreator
from dataloading.loadOriginalSize import getDatasetFromDirectory
import numpy as np
import time
import keras
import pandas as pd
from matplotlib import pyplot as plt


batch_size = 1

sceneContentApproximator = keras.models.load_model(
    'baseline_model/saved_weights/scene_content_approximator.keras',
    custom_objects={'SceneContentApproximator': SceneContentApproximator}
)

# Get the number of channels from the loaded model
num_kernels = sceneContentApproximator.conv.filters
selfDescriptionCreator = SelfDescriptionCreator(11, 11, num_kernels, learning_rate=0.1)

label_list = []

folder_list = ["ProGAN"]
for folder in folder_list:

    if os.path.isdir(os.path.join('datasets', folder)):

    
        print(folder)
        dataset = getDatasetFromDirectory(
            os.path.join('datasets', folder), 
            batch_size=batch_size, 
        )

        tf.concat
        os.makedirs(os.path.join('baseline_model/self_descriptions', folder), exist_ok=True)
        os.makedirs(os.path.join('baseline_model/self_descriptions', folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join('baseline_model/self_descriptions', folder, 'test'), exist_ok=True)

        i = 0
        batch_counter
        batch_size = 32
        selfdescriptions_list = []
        start_time = time.time()
        for image_batch in dataset:
            

            approximations = sceneContentApproximator(image_batch)

            residuals = image_batch - approximations

            selfdescription = selfDescriptionCreator.train_and_get(residuals, 10000)
            batch_counter += 1
            selfdescriptions_list.append(selfdescription)

            if batch_counter >= batch_size:
                selfdescription_batch = tf.stack(selfdescriptions_list, axis=0)
                selfdescriptions_list = []
                if np.random.rand() < 0.2:
                    subfolder = 'test'
                else:
                    subfolder = 'train'
                    np.save(os.path.join('baseline_model/self_descriptions', folder, subfolder, f'batch_{i}.npy'), selfdescription_batch.numpy())

                current_time = time.time()
                tf.print(f"Processed batch {i} in {current_time - start_time:.2f} seconds")
                start_time = current_time
                i += 1
                batch_counter = 0

            