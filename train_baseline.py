import os
import numpy as np
import tensorflow as tf
from dataloading.loadResized import getDatasetFromDirectory, getLabeledDatasetFromDirectory
from baseline_model.SceneContentApproximator import SceneContentApproximator
from baseline_model.SelfDescriptionCreator import SelfDescriptionCreator
from baseline_model.Attributor import Attributor


resize_height, resize_width = 256, 256

# should be odd numbers
kernel_height, kernel_width = 11, 11
# in the paper detoned as K
num_kernels = 4

learning_rate = 0.001

# please refer to the paper
loss_constant_alpha = 0.01

# please refer to the paper 
loss_constant_lambda = 1.0

sceneContentApproximator_epochs = 1

selfdescriptionCreator_epochs = 10

num_components = 3

batch_size = 32



real_dataset = getDatasetFromDirectory(os.path.join('datasets', 'coco2017'), batch_size)

real_dataset = real_dataset.take(512)

sceneContentApproximator = SceneContentApproximator(num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda)

if os.path.exists(os.path.join('model', 'sceneContentApproximator.weights.h5')) and False:
    # load the model
    sceneContentApproximator.build(input_shape=(batch_size, resize_height, resize_width, 1)) 
    sceneContentApproximator.load_weights(os.path.join('model', 'sceneContentApproximator.weights.h5'))
else:
    # train sceneContentApproximator only with real images
    sceneContentApproximator.train(real_dataset, sceneContentApproximator_epochs)
    # save the model
    sceneContentApproximator.save_weights(os.path.join('model', 'sceneContentApproximator.weights.h5'))


labeled_dataset = getLabeledDatasetFromDirectory('datasets', batch_size)

labeled_dataset = labeled_dataset.take(2)

selfdescriptionCreator = SelfDescriptionCreator(kernel_height, kernel_width, num_components, learning_rate)

all_labels = []
all_selfdescriptions = []


for image_batch, labels in labeled_dataset:
    residuals = sceneContentApproximator(image_batch)
    selfdescriptions = selfdescriptionCreator.train_and_get(residuals, selfdescriptionCreator_epochs)
    all_labels.append(labels)
    all_selfdescriptions.append(selfdescriptions)

all_labels = tf.concat(all_labels, axis = 0)
all_selfdescriptions = tf.concat(all_selfdescriptions, axis = 0)



attributor = Attributor(all_selfdescriptions.numpy(), np.squeeze(all_labels.numpy()), 3)


predictions = attributor.predict(all_selfdescriptions.numpy(), 0.1)
print(predictions)
