import os
import numpy as np
import tensorflow as tf
from pathlib import Path

from baseline_model.SceneContentApproximator import SceneContentApproximator
from baseline_model.SelfDescriptionCreator import SelfDescriptionCreator
from dataloading.loadResized import getDatasetFromDirectory


def setup_gpu():
    """Configure GPU memory growth to avoid allocation issues."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")


def process_dataset(dataset_path, output_dir, batch_size=32, epochs=10):
    dataset = getDatasetFromDirectory(
        dataset_path, 
        batch_size=batch_size,
        resize_height=256,
        resize_width=256
    )
    
    sceneContentApproximator = tf.keras.models.load_model(
    'baseline_model/scene_content_approximator.keras',
    custom_objects={'SceneContentApproximator': SceneContentApproximator}
)
    
    selfdescription_creator = SelfDescriptionCreator(
        kernel_height=11,
        kernel_width=11,
        L=3,
        learning_rate=0.001
    )

    all_selfdescriptions = []

    for image_batch in dataset:
        # Generate residuals
        residuals = image_batch - sceneContentApproximator(image_batch)

        # Create self-descriptions
        selfdescriptions = SelfDescriptionCreator(11, 11, 3, 0.1).train_and_get(residuals, epochs=epochs)
        all_selfdescriptions.append(selfdescriptions.numpy())


    final_selfdescriptions = np.concatenate(all_selfdescriptions, axis=0)
    
    # Save to output directory
    dataset_name = Path(dataset_path).name
    output_file = Path(output_dir) / f"{dataset_name}_selfdescriptions.npy"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_file, final_selfdescriptions)
    print(f"Saved {final_selfdescriptions.shape[0]} self-descriptions to {output_file}")
    
    return output_file




if __name__ == "__main__":
    setup_gpu()
    
    for folder in os.listdir('datasets/OSMA/fake'):
        if os.path.isdir(os.path.join('datasets/OSMA/fake', folder)):
            dataset_path = os.path.join('datasets/OSMA/fake', folder)
            output_dir = 'datasets/self_descriptions'
            print(f"Processing dataset: {dataset_path}")
            process_dataset(dataset_path, output_dir, batch_size=32, epochs=10000)