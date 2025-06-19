import os
import sys
project_root = os.environ.get('BACHELOR_MODELS_ROOT', '.')
sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt 
from baseline_model.source.Attributor import Attributor
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
import seaborn as sns

TAU              = 1500                         
UNKNOWN_LABEL    = -1                                    
attr = Attributor.load("baseline_model/saved_weights/attributor")

# thats the folders the attributor was trained on
folder_names = ["OSMA_AttGAN", "OSMA_BigGAN", "OSMA_FaceSwap","OSMA_FSGAN", "OSMA_InfoMax-GAN", "OSMA_MMDGAN", "OSMA_ProGAN", "OSMA_S3GAN", "OSMA_SAGAN", "OSMA_SNGAN", "OSMA_styleGAN2", "imagenet_glide", "imagenet_wukong", "imagenet_vqdm"]

X_list: list[np.ndarray] = []
Y_list: list[np.ndarray] = []

i = 0

for folder in folder_names:
    for batch in os.listdir(os.path.join('baseline_model/self_descriptions', folder, 'test')):
        batch_path = os.path.join('baseline_model/self_descriptions', folder, 'test', batch)
        X = np.load(batch_path, allow_pickle=False)
        X_list.append(X)
        Y_list.append(np.full((X.shape[0],), i, dtype=np.int32))
    i += 1


for folder in os.listdir(os.path.join('baseline_model/self_descriptions')):
    if os.path.isdir(os.path.join('baseline_model/self_descriptions', folder)) and folder not in folder_names:
        for batch in os.listdir(os.path.join('baseline_model/self_descriptions', folder, 'test')):
            batch_path = os.path.join('baseline_model/self_descriptions', folder, 'test', batch)
            X = np.load(batch_path, allow_pickle=False)
            X_list.append(X)
            Y_list.append(np.full((X.shape[0],), -1, dtype=np.int32))

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)

Y_pred = attr.predict(X, tau=TAU)

ID2NAME = {-1: "unknown"}
ID2NAME.update({i: n for i, n in enumerate(folder_names)})
LABELS = [-1] + list(range(len(folder_names)))   # [-1,0,1,…,9]
NAME_ORDER = ["unknown"] + folder_names 


acc  = accuracy_score(Y, Y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    Y, Y_pred, labels=LABELS, average="macro", zero_division=0
)


cm = confusion_matrix(Y, Y_pred, labels=LABELS)

cm_df = pd.DataFrame(cm, index=NAME_ORDER, columns=NAME_ORDER)
print("\nConfusion matrix (rows = true, cols = pred):")
print(cm_df)

print("Accuracy  :", f"{acc:.4f}")
print("Precision :", f"{prec:.4f}")
print("Recall    :", f"{rec:.4f}")
print("F1‑score  :", f"{f1:.4f}")

metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
with open("baseline_model/evaluation/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics -> baseline_model/evaluation/results/metrics.json")

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("baseline_model/evaluation/results/confusion_matrix.png", dpi=200)
print("Saved heat-map -> baseline_model/evaluation/results/confusion_matrix.png")
