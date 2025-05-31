from dataloading.loadResized import getDatasetFromDirectory
from baseline_model.SceneContentApproximator import SceneContentApproximator
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Your existing code here

coco = getDatasetFromDirectory('datasets/coco2017', 32)
imagenet = getDatasetFromDirectory('datasets/imagenet/train', 32)


coco.concatenate(imagenet)

num_kernels = 8
kernel_height, kernel_width = 11, 11
learning_rate = 0.001
loss_constant_alpha = 0.01
loss_constant_lambda = 1.0


sceneContentApproximator = SceneContentApproximator(num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda)
sceneContentApproximator.train(coco, 10)

sceneContentApproximator.save("baseline_model/scene_content_approximator.h5")
sceneContentApproximator.save("baseline_model/scene_content_approximator.keras")

print("Training complete and model saved.")