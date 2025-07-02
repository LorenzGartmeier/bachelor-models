import tensorflow as tf
import numpy as np
import os, sys, pathlib, random
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '16384'
project_root = os.environ.get("BACHELOR_MODELS_ROOT", ".")
sys.path.append(project_root)

from bayesian_model.source.SceneContentApproximator import SceneContentApproximator
from bayesian_model.source.SelfDescriptionCreator   import SelfDescriptionCreator
from dataloading.loadOriginalSize import getDatasetFromDirectory

MODEL_PATH = "bayesian_model/saved_weights/scene_content_approximator.keras"
OUT_ROOT   = "bayesian_model/self_descriptions"
DATA_ROOT  = "datasets"
EXCLUDE_FOLDERS = ["imagenet", "coco2017"]     # real images      

num_batches = 500
num_test_samples = 14
EPOCHS_PER_IMAGE = 2000              

for g in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

sceneContentApproximator = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"SceneContentApproximator": SceneContentApproximator})

num_kernels = sceneContentApproximator.num_kernels
creator     = SelfDescriptionCreator(11, 11, num_kernels,
                                     L=3, learning_rate=0.05)

for folder in os.listdir(DATA_ROOT):

    if folder in EXCLUDE_FOLDERS: continue
    ds_path = pathlib.Path(DATA_ROOT, folder)

    out_train = pathlib.Path(OUT_ROOT, folder, "train");  out_train.mkdir(parents=True, exist_ok=True)
    out_test  = pathlib.Path(OUT_ROOT, folder, "test");   out_test.mkdir(parents=True,  exist_ok=True)

    dataset = getDatasetFromDirectory(str(ds_path), batch_size=1).take(num_batches)

    batch_idx = 0
    batch_size = 32
    batch_counter = 0
    test_idx = random.sample(range(0, num_batches), int(num_batches*0.2))
    for img_batch in dataset.take(70): 



        if batch_idx in test_idx:
            test_folder = pathlib.Path(OUT_ROOT, folder, "test/batch" + str(batch_idx))
            test_folder.mkdir(parents=True, exist_ok=True)
            for i, sample in enumerate(sceneContentApproximator.predict(img_batch, n_samples=num_test_samples)):
                residual_sample = img_batch - sample
                selfdescription_sample = creator.train_and_get(residual_sample, epochs=EPOCHS_PER_IMAGE)
                np.save(test_folder / f"sample{i}.npy", selfdescription_sample.numpy())
        else:
            approximation_batch = sceneContentApproximator.predict(img_batch, n_samples=1)[0]
            residual_batch = img_batch - approximation_batch
            selfdescription_batch = creator.train_and_get(residual_batch, epochs=EPOCHS_PER_IMAGE)
            np.save(out_train / f"batch{batch_idx}.npy", selfdescription_batch.numpy())

        batch_idx += 1



