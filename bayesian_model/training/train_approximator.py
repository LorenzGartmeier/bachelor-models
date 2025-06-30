import sys
import os

project_root = os.environ.get('BACHELOR_MODELS_ROOT', '.')
sys.path.append(project_root)
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from dataloading.loadOriginalSize import getDatasetFromDirectory
from bayesian_model.source.SceneContentApproximator import SceneContentApproximator


gpus = tf.config.experimental.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)


batch_size   = 64
coco    = getDatasetFromDirectory("datasets/coco2017", 1).take(100000)
imagenet= getDatasetFromDirectory("datasets/imagenet", 1).take(200000)
dataset = coco.concatenate(imagenet)


num_kernels           = 8
kernel_height         = 11
kernel_width          = 11
learning_rate         = 0.001
loss_constant_alpha   = 1e-2
loss_constant_lambda  = 0.5        

model = SceneContentApproximator(
            num_kernels          = num_kernels,
            kernel_height        = kernel_height,
            kernel_width         = kernel_width,
            learning_rate        = learning_rate,
            loss_constant_alpha  = loss_constant_alpha,
            loss_constant_lambda = loss_constant_lambda)   


history = model.train(dataset, epochs = 10)   

model.save("bayesian_model/saved_weights/scene_content_approximator.keras")



hist_df = pd.DataFrame(history)                 # recon, diversity, KL, total
csv_path =  "bayesian_model/training/results/approximator_training_history.csv"
hist_df.to_csv(csv_path, index=False)



plt.figure(figsize=(12, 6))
for k, v in history.items():
    plt.plot(v, label=k, linewidth=2)
plt.title("Bayesian Scene-Content Approximator – training losses")
plt.xlabel("Epoch");  plt.ylabel("Loss")
plt.legend();  plt.grid(alpha=.3);  plt.tight_layout()

png_path = "bayesian_model/training/results/approximator_training_losses.png"
plt.savefig(png_path, dpi=300)
plt.close()
