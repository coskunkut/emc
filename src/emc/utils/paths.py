import os
from pathlib import Path

from natsort import natsorted

def get_paths():
	paths = {
		"root": os.path.abspath(Path(__file__).parent.parent.parent.parent)
	}
	paths["exps"] = os.path.join(paths["root"], "exps")
	paths["default_params"] = os.path.join(paths["root"], "src", "emc", "resources", "default_params")
	paths["config"] = {
		"mpl": os.path.join(paths["root"], "src", "emc", "resources", "mpl")
	}
	paths["data"] = {
		"realworld_har": {
			"root": os.path.join(paths["root"], "data", "realworld_har"),
			"samples": os.path.join(paths["root"], "data", "realworld_har", "prepared"),
		},
		"cwru": {
			"root": os.path.join(paths["root"], "data", "cwru"),
			"samples": os.path.join(paths["root"], "data", "cwru", "samples"),
			"info": os.path.join(paths["root"], "exps", "cwru", "info.csv")
		},
		"eeg_eye_state": {
			"root": os.path.join(paths["root"], "data", "eeg_eye_state")
		}
	}
	return paths
