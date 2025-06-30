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

coco = getDatasetFromDirectory('datasets/coco2017', 64).take(512)




num_kernels = 8
kernel_height, kernel_width = 11, 11
learning_rate = 0.0001
loss_constant_alpha = 0.01
loss_constant_lambda = 1.0

sceneContentApproximator = SceneContentApproximator(num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda)
history, kernel_history, singular_values_history = sceneContentApproximator.train(coco, 1)

sceneContentApproximator.save("baseline_model/saved_weights/scene_content_approximator.keras")


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



# Save training history
kernel_history_df = pd.DataFrame(kernel_history)
kernel_history_df.to_csv("baseline_model/training/results/attributor_kernel_history.csv", index=False)

    
plt.figure(figsize=(12, 6))
for kernel in kernel_history:
    plt.plot(kernel, linewidth=0.5)

plt.title("kernel values", fontsize=14)
plt.xlabel("batch", fontsize=12)
plt.ylabel("kernel values", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_model/training/results/attributor_kernel_values.png", dpi=300)
plt.close()

# Save training history
singular_values_history_df = pd.DataFrame(singular_values_history)
singular_values_history_df.to_csv("baseline_model/training/results/attributor_singualar_values_history.csv", index=False)

    
plt.figure(figsize=(12, 6))
for singular_value in singular_values_history:
    plt.plot(singular_value, linewidth=0.5)

plt.title("singular values", fontsize=14)
plt.xlabel("batch", fontsize=12)
plt.ylabel("singular values", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_model/training/results/attributor_singular_values.png", dpi=300)
plt.close()

print("Training complete and model saved.")