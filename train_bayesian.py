import os
import numpy as np
import tensorflow as tf
from dataloading.loadResized import getDatasetFromDirectory, getLabeledDatasetFromDirectory
from bayesian_model.SceneContentApproximator import SceneContentApproximator
from bayesian_model.SelfDescriptionCreator import SelfDescriptionCreator
from bayesian_model.Attributor import Attributor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



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

num_components = 3

batch_size = 32

resize_height = 128
resize_width = 128



real_dataset = getDatasetFromDirectory(os.path.join('datasets', 'coco2017',), batch_size, resize_height, resize_width)

real_dataset = real_dataset.take(1024)

sceneContentApproximator = SceneContentApproximator(num_kernels, kernel_height, kernel_width, learning_rate, loss_constant_alpha, loss_constant_lambda)

if os.path.exists(os.path.join('bayesian_model', 'sceneContentApproximator.weights.h5')):
    # load the model
    sceneContentApproximator.build(input_shape=(batch_size, resize_height, resize_width, 1)) 
    sceneContentApproximator.load_weights(os.path.join('bayesian_model', 'sceneContentApproximator.weights.h5'))
else:
    # train sceneContentApproximator only with real images
    sceneContentApproximator.train(real_dataset, sceneContentApproximator_epochs)
    # save the model
    sceneContentApproximator.save_weights(os.path.join('bayesian_model', 'sceneContentApproximator.weights.h5'))


labeled_dataset = getLabeledDatasetFromDirectory(os.path.join('datasets', 'fake'), batch_size, resize_height, resize_width)

selfdescriptionCreator = SelfDescriptionCreator(kernel_height, kernel_width, num_components, learning_rate)

all_labels = []
all_selfdescriptions = []


for image_batch, labels in labeled_dataset:
    residuals = sceneContentApproximator(image_batch)
    selfdescriptions = selfdescriptionCreator.train_and_get(residuals, selfdescriptionCreator_epochs)
    all_labels.append(labels)
    all_selfdescriptions.append(selfdescriptions)
    print("batch prediction done")

all_labels = tf.concat(all_labels, axis = 0)
all_selfdescriptions = tf.concat(all_selfdescriptions, axis = 0)



attributor = Attributor(all_selfdescriptions.numpy(), np.squeeze(all_labels.numpy()), 3)


predictions = attributor.predict(all_selfdescriptions.numpy(), 0.1)



# Convert tensors/arrays to numpy if needed
y_true = np.squeeze(all_labels.numpy())
y_pred = predictions

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision, Recall, F1 (for binary or multiclass)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)