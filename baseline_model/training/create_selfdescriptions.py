# Expects a trained and saved SceneContentApproximator in baseline_model/scene_content_approximator.keras
import sys
import os
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '20000'  # Limit the workspace to 20GB
project_root = os.environ.get('BACHELOR_MODELS_ROOT', '.')
sys.path.append(project_root)

import tensorflow as tf
from baseline_model.source.SceneContentApproximator import SceneContentApproximator
from baseline_model.source.SelfDescriptionCreator import SelfDescriptionCreator
from dataloading.loadOriginalSize import getDatasetFromDirectory
import numpy as np
import time
import keras


sceneContentApproximator = keras.models.load_model(
    'baseline_model/saved_weights/scene_content_approximator.keras',
    custom_objects={'SceneContentApproximator': SceneContentApproximator}
)

# Get the number of channels from the loaded model
num_kernels = sceneContentApproximator.conv.filters
selfDescriptionCreator = SelfDescriptionCreator(11, 11, num_kernels, learning_rate=0.1)

label_list = []

exclusion_list = ["coco2017", "imagenet", "starGAN", "styleGAN", "wav2lip", "styleGAN2", "SNGAN", "SAGAN", "imagenet_vqdm", "imagenet_wukong", "MMDGAN", "FSGAN", "FaceSwap"]
for folder in os.listdir('datasets'):

    if os.path.isdir(os.path.join('datasets', folder)) and folder not in exclusion_list:

    
        print(folder)
        dataset = getDatasetFromDirectory(
            os.path.join('datasets', folder), 
            batch_size = 1, 
        ).take(5000)

        os.makedirs(os.path.join('baseline_model/self_descriptions', folder), exist_ok=True)
        os.makedirs(os.path.join('baseline_model/self_descriptions', folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join('baseline_model/self_descriptions', folder, 'test'), exist_ok=True)

        i = 0
        batch_counter = 0
        batch_size = 32
        selfdescriptions_list = []
        start_time = time.time()
        for image_batch in dataset:
            tf.print("Processing batch...")

            approximations = sceneContentApproximator(image_batch)

            residuals = image_batch - approximations

            tf.print("residuals created")

            selfdescription = selfDescriptionCreator.train_and_get(residuals, tf.constant(2000, tf.int32))
            batch_counter += 1
            selfdescriptions_list.append(selfdescription)

            tf.print("selfdescription created")

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

            