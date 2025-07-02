
import os, sys, json, re, math, pathlib, collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

TAU_PROB = 0.55            #
TAU_UNC  = math.log(10)   

ATTR_PATH = "bayesian_model/saved_weights/attributor"
ROOT      = "bayesian_model/self_desciptions"

folder_names = ["ProGAN", "SNGAN"]

sys.path.append(os.environ.get("BACHELOR_MODELS_ROOT", "."))
from baseline_model.source.Attributor import Attributor 
attr = Attributor.load(ATTR_PATH)

batch_re = re.compile(r"batch_(\\d+)")
sample_re= re.compile(r"sample_(\\d+)")

def load_batches(source_path):
    true_batches = []
    for batch_dir in sorted(pathlib.Path(source_path).glob("batch_*")):
        samples = []
        for npy in sorted(batch_dir.glob("sample_*.npy")):
            samples.append(np.load(npy, allow_pickle=False))
        if samples:
            true_batches.append(samples)
    return true_batches

X_lists, Y_true = [], []

for idx, folder in enumerate(folder_names):
    sub = pathlib.Path(ROOT, folder, "test")
    if not sub.exists(): continue
    for samples in load_batches(sub):
        X_lists.append(samples)
        Y_true.append(np.full(samples[0].shape[0], idx, np.int32))

for folder in pathlib.Path(ROOT).glob("*"):
    if folder.name not in folder_names and (folder / "test").is_dir():
        for samples in load_batches(folder / "test"):
            X_lists.append(samples)
            Y_true.append(np.full(samples[0].shape[0], -1, np.int32))

Y = np.concatenate(Y_true, axis=0)

labels_all, entropy_all = [], []
for samples in X_lists:                          
    lbl, ent = attr.predict(samples,
                            tau_prob=TAU_PROB,
                            tau_unc =TAU_UNC)
    labels_all.append(lbl)        # (B,)
    entropy_all.append(ent)       # (B,)

Y_pred = np.concatenate(labels_all, axis=0)

ID2NAME = {-1: "unknown"}
ID2NAME.update({i: n for i, n in enumerate(folder_names)})

LABELS      = [-1] + list(range(len(folder_names)))
NAME_ORDER  = ["unknown"] + folder_names

acc  = accuracy_score(Y, Y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    Y, Y_pred, labels=LABELS, average="macro", zero_division=0
)

cm = confusion_matrix(Y, Y_pred, labels=LABELS)
cm_df = pd.DataFrame(cm, index=NAME_ORDER, columns=NAME_ORDER)

print("\\nConfusion matrix (rows = true, cols = pred):")
print(cm_df)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

out_dir = pathlib.Path("bayesian_model/evaluation/results")
out_dir.mkdir(parents=True, exist_ok=True)

cm_df.to_csv(out_dir / "confusion_matrix.csv")
with open(out_dir / "metrics.json", "w") as f:
    json.dump(dict(accuracy=acc, precision=prec,
                   recall=rec, f1=f1), f, indent=2)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.ylabel("True label"); plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(out_dir / "confusion_matrix.png", dpi=250)
print(f"Saved results → {out_dir}")
