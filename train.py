import os

import tensorflow as tf
from dataloading.loadResized import getDatasetFromDirectory, getLabeledDatasetFromDirectory
from model import Attributor, SelfDescriptionCreator
from model.SceneContentApproximator import SceneContentApproximator


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

sceneContentApproximator_epochs = 10

selfdescriptionCreator_epochs = 10

n_components = 3



real_dataset = getDatasetFromDirectory(os.path.join('datasets', 'train2017'), 32)

sceneContentApproximator = SceneContentApproximator(num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda)

if os.path.exists(os.path.join('model', 'sceneContentApproximator.h5')):
    # load the model
    sceneContentApproximator.load_weights(os.path.join('model', 'sceneContentApproximator.h5'))
else:
    # train sceneContentApproximator only with real images
    sceneContentApproximator.train(real_dataset, sceneContentApproximator_epochs)
    # save the model
    sceneContentApproximator.save_weights(os.path.join('model', 'sceneContentApproximator.h5'))


labeled_dataset = getLabeledDatasetFromDirectory('datasets', 32)

selfdescriptionCreator = SelfDescriptionCreator(kernel_height, kernel_width)

all_labels = tf.constant([], dtype=tf.int32)
all_residuals = tf.constant([], dtype=tf.float32)


for image_batch, labels in labeled_dataset:
    resiuals = sceneContentApproximator(image_batch)
    selfdescriptions = selfdescriptionCreator.train_and_get(resiuals, selfdescriptionCreator_epochs)
    all_labels = tf.concat([all_labels, labels], axis=0)
    all_residuals = tf.concat([all_residuals, selfdescriptions], axis=0)

attributor = Attributor(all_labels, all_residuals, 3)










