# Expects a trained and saved SceneContentApproximator in baseline_model/scene_content_approximator.keras
import sys
import os

project_root = os.environ.get('BACHELOR_MODELS_ROOT', '.')
sys.path.append(project_root)

import tensorflow as tf
from baseline_model.source.SceneContentApproximator import SceneContentApproximator
from baseline_model.source.SelfDescriptionCreator import SelfDescriptionCreator
from dataloading.loadCropped import getDatasetFromDirectory
import numpy as np


batch_size = 32

sceneContentApproximator = tf.keras.models.load_model(
    'baseline_model/saved_weights/scene_content_approximator.keras',
    custom_objects={'SceneContentApproximator': SceneContentApproximator}
)

# Get the number of channels from the loaded model
num_kernels = sceneContentApproximator.conv.filters
selfDescriptionCreator = SelfDescriptionCreator(11, 11, num_kernels, L = 3, learning_rate=0.1, beta=0.001)

label_list = []

folder_list = ["imagenet_adm"]
for folder in folder_list:

    if os.path.isdir(os.path.join('datasets', folder)):

    
        print(folder)
        dataset = getDatasetFromDirectory(
            os.path.join('datasets', folder), 
            batch_size=batch_size, 
        )

        os.makedirs(os.path.join('baseline_model/self_descriptions', folder), exist_ok=True)
        os.makedirs(os.path.join('baseline_model/self_descriptions', folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join('baseline_model/self_descriptions', folder, 'test'), exist_ok=True)

        i = 0
        for image_batch in dataset:
            if os.path.exists(os.path.join('baseline_model/self_descriptions', folder, f'train/batch_{i}.npy')) \
            or os.path.exists(os.path.join('baseline_model/self_descriptions', folder, f'test/batch_{i}.npy')):
                i += 1
                continue

            if i > 500:
                break


            approximations = sceneContentApproximator(image_batch)

            residuals = image_batch - approximations
            selfdescriptions, losses = selfDescriptionCreator.train_and_get(residuals, 500)
            tf.print(selfdescriptions.numpy(), losses.numpy().mean())


            if np.random.rand() < 0.2:
                 subfolder = 'test'
            else:
                subfolder = 'train'
            np.save(os.path.join('baseline_model/self_descriptions', folder, subfolder, f'batch_{i}.npy'), selfdescriptions.numpy())

            i += 1