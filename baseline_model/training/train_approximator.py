import sys
import os

project_root = os.environ.get('BACHELOR_MODELS_ROOT', '.')
sys.path.append(project_root)

from dataloading.loadCropped import getDatasetFromDirectory
from baseline_model.source.SceneContentApproximator import SceneContentApproximator
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt 

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

coco = getDatasetFromDirectory('datasets/coco2017', 64)
imagenet = getDatasetFromDirectory('datasets/imagenet/test', 64)


coco = coco.concatenate(imagenet).take(2048)

num_kernels = 8
kernel_height, kernel_width = 11, 11
learning_rate = 0.0001
loss_constant_alpha = 0.01
loss_constant_lambda = 0.1


sceneContentApproximator = SceneContentApproximator(num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda)
history = sceneContentApproximator.train(coco, 10)

sceneContentApproximator.save("baseline_model/saved_weights/scene_content_approximator.keras")


weights = sceneContentApproximator.get_weights()[0]
weights_max =tf.reduce_max(weights).numpy()
weights_min = tf.reduce_min(weights).numpy()

print(f"Kernel weights range: {weights_min} to {weights_max}")

# Save training history
history_df = pd.DataFrame(history)
history_df.to_csv("baseline_model/training/results/attributor_training_history.csv", index=False)
    
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
plt.savefig("baseline_model/training/results/attributor_training_losses.png", dpi=300)
plt.close()

print("Training complete and model saved.")