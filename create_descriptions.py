import tensorflow as tf 
import numpy as np
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
        selfdescriptions = selfdescription_creator.train_and_get(residuals, epochs=epochs)
        all_selfdescriptions.append(selfdescriptions.numpy())

    final_selfdescriptions = np.concatenate(all_selfdescriptions, axis=0)
    
    # Save to output directory
    dataset_name = Path(dataset_path).name
    output_file = Path(output_dir) / f"{dataset_name}_selfdescriptions.npy"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_file, final_selfdescriptions)

if __name__ == "__main__":
    setup_gpu()

    for folder in Path('datasets').iterdir():
        if folder.is_dir():
            dataset_path = str(folder)
            output_dir = 'self_descriptions'
            print(f"Processing dataset: {dataset_path}")
            process_dataset(dataset_path, output_dir, batch_size=32, epochs=10)
            print(f"Self-descriptions saved to: {output_dir}/{folder.name}_selfdescriptions.npy")