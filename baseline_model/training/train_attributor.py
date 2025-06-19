import sys
import os

project_root = os.environ.get('BACHELOR_MODELS_ROOT', '.')
sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt 
from baseline_model.source.Attributor import Attributor


def get_dataset(path):
    def load_npy_batch(file_path):
        data = tf.numpy_function(
            func=lambda x: np.load(x, allow_pickle=False),  
            inp=[file_path],
            Tout=tf.float32 
        )
        return data
    dataset = tf.data.Dataset.list_files(path, shuffle=True)
    dataset = dataset.map(
        load_npy_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset


attributor = Attributor(num_components=16)
i = 0

folder_names = ["OSMA/AttGAN", "OSMA/BigGAN", "OSMA/FaceSwap","OSMA/FSGAN", "OSMA/InfoMax-GAN", "OSMA/MMDGAN", "OSMA/ProGAN", "OSMA/S3GAN", "OSMA/SAGAN", "OSMA/SNGAN", "OSMA/styleGAN2", "imagenet_glide", "imagenet_wukong", "imagenet_vqdm"]

for folder in folder_names:

    print(f"Training GMM for folder: {folder}")
    dataset = get_dataset(os.path.join('baseline_model/self_descriptions', folder, 'train/*.npy'))

    history = attributor.add_gmm(i, dataset, epochs=20, lr=1e-2)

    folder = folder.replace("/", "_")
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"baseline_model/training/results/gmm_loss_{folder}.csv", index=False)

    # Generate and save loss plot
    plt.figure(figsize=(12, 6))
    for loss_type in history:
        plt.plot(history[loss_type], label=loss_type, linewidth=2)

    plt.title("Training Losses", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"baseline_model/training/results/gmm_loss_{folder}.png", dpi=300)
    plt.close()

    i += 1

attributor.save('baseline_model/saved_weights/attributor')
print("Training complete and model saved.")