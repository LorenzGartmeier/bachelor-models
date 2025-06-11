from dataloading.loadOriginalSize import getDatasetFromDirectory
from baseline_model.SceneContentApproximator import SceneContentApproximator
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Your existing code here

coco = getDatasetFromDirectory('datasets/coco2017', 128)
imagenet = getDatasetFromDirectory('datasets/imagenet/test', 128)


coco.concatenate(imagenet)

num_kernels = 8
kernel_height, kernel_width = 11, 11
learning_rate = 0.001
loss_constant_alpha = 0.01
loss_constant_lambda = 1.0


sceneContentApproximator = SceneContentApproximator(num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda)
history = sceneContentApproximator.train(coco, 10)

sceneContentApproximator.save("baseline_model/scene_content_approximator.h5")
sceneContentApproximator.save("baseline_model/scene_content_approximator.keras")

# Save training history
history_df = pd.DataFrame(history)
history_df.to_csv("baseline_model/training_history.csv", index=False)

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
plt.savefig("baseline_model/training_losses.png", dpi=300)
plt.close()

print("Training complete and model saved.")