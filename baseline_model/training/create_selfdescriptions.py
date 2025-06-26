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
import time
import pandas as pd
from matplotlib import pyplot as plt


batch_size = 32

sceneContentApproximator = tf.keras.models.load_model(
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

        start_time = time.time()
        for image_batch in dataset:
            if os.path.exists(os.path.join('baseline_model/self_descriptions', folder, f'train/batch_{i}.npy')) \
            or os.path.exists(os.path.join('baseline_model/self_descriptions', folder, f'test/batch_{i}.npy')):
                i += 1
                continue

            if i > 500:
                break


            approximations = sceneContentApproximator(image_batch)

            residuals = image_batch - approximations

            selfdescriptions, loss_list = selfDescriptionCreator.train_and_get(residuals, 4000)
            current_time = time.time()
            tf.print(f"Processed batch {i} in {current_time - start_time:.2f} seconds")
            start_time = current_time



                        # Save training history
            history_df = pd.DataFrame(loss_list)
            history_df.to_csv(f"baseline_model/training/results/selfdescriptions_loss_batch_{i}.csv", index=False)
                
            # Generate and save loss plot
            plt.figure(figsize=(12, 6))
            for sample_loss in loss_list:
                plt.plot(sample_loss, label=None, linewidth=2)

            plt.title("Training Losses", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"baseline_model/training/results/selfdescriptions_loss_batch_{i}.png", dpi=300)
            plt.close()


            if np.random.rand() < 0.2:
                 subfolder = 'test'
            else:
                subfolder = 'train'
            np.save(os.path.join('baseline_model/self_descriptions', folder, subfolder, f'batch_{i}.npy'), selfdescriptions.numpy())

            i += 1

            