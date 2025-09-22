import os
import pickle
from pathlib import Path

import mne
import pandas as pd
from pycrostates.cluster import ModKMeans
from scipy.io import arff

from emc.utils.paths import get_paths

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "eeg", "output")
Path(output_dir_path).mkdir(parents=True, exist_ok=True)

# load data
file_path = os.path.join(paths["data"]["eeg_eye_state"]["root"], "EEG Eye State.arff")
data = arff.loadarff(file_path)
df = pd.DataFrame(data[0])

Y = df["eyeDetection"].to_numpy().flatten().astype(int)
X = df.drop(["eyeDetection"], axis=1)

# MNE
ch_names = X.columns.tolist()
info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types="eeg")
X_mne = mne.io.RawArray(X.transpose(), info)

# get microstates
n_clusters = 8
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
ModK.fit(X_mne, n_jobs=8)
segmentation = ModK.predict(X_mne, factor=10)
ms_seq = segmentation.labels

# save data
data = {
    "meta": {
        "n_clusters": n_clusters,
        "ch_names": ch_names
    },
    "X": X_mne.get_data().T,
    "ms_seq": ms_seq,
    "Y": Y
}
with open(os.path.join(output_dir_path, "data.pkl"), "wb") as file:
    pickle.dump(data, file)
print(f"data saved: {output_dir_path}/data.pkl")
