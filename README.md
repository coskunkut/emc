# Evolving Markov Chains

## Updates

Update (Revision 2025-09): The revised implementation with reduced time complexity is available in the `rev1` branch.

## Introduction

Evolving Markov Chains (EMC) is an online method for mode discovery and recognition.
This repository contains the implementation in `Python` and the necessary utilities to evaluate results.

The experiments contain:
- synthetic data (probability tracking, change point detection, mode discovery and recognition)
- human activity recognition through accelerometer readings
- electric motor condition monitoring through vibration data
- eye-state detection through EEG signals

## Installation

To install the `emc` package, you can use `conda`:

```sh
conda env create -f environment.yaml -n emc
conda activate emc
pip install -e .
```

## Usage

Here is how to run EMC on a `sequence`:

```python
from emc.estimator.EMC import EMC

emc_ins = EMC(
    alpha=4,
    order=1,
    lambda_=[0.94, 0.98],
    beta=0,
    delta=[0.3, 0.2],
    eta=[0.4, 0.3],
    tau=50,
)
emc_ins.process_sequence(sequence, progress=True)
```

Synthetic data generation

```python
from emc.generator.markov import MarkovianSwitchingSystem

mss_ins = MarkovianSwitchingSystem(
    alphabet_cardinality=4,
    mode_process_order=1,
    subprocess_order=1,
    number_of_subprocesses=5,
    rng_or_seed=42
)
mss_ins.generate_subprocess_sequence(
    number_of_regimes=10,
    avoid_loops=True
)
symbol_sequence = mss_ins.generate_symbol_sequence(
    regime_length_bounds=[500,1000]
)
```

## Reproducing Experiment Results

Source code for experiments is located in `exps` directory with the following structure:
- synthetic
    - 01_pt: probability tracking
    - 02_cpd: change point detection
    - 03_mdr: mode discovery and recognition
- har: human activity recognition on `RealWorld (HAR)` dataset ([link](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/))
- cwru: electric motor condition monitoring and anomaly detection on `CWRU` dataset ([link](https://engineering.case.edu/bearingdatacenter))
- eeg: eye-state detection on EEG signals ([link](https://doi.org/10.24432/C57G7J))

Each directory contains the necessary steps to get the results shown in the paper.

### Experiments on Synthetic Data

The data is not included in the repo but can be generated using the `generate_data.py` file.
```python
python exps/synthetic/01_pt/generate_data.py
```

The second step is to run hyperparameter optimization.
The values used in the paper are included in `/output/params.json` file.
Therefore this step can be skipped (optimization takes some time).
If needed, it can be run again by:
```python
python exps/synthetic/01_pt/optimize_params.py
```

Finally, using the data and the parameter values, the test runs are executed as follows:
```python
python exps/synthetic/01_pt/test.py
```

The resuls will be saved to `output/results` directory.

### Human Activity Recognition Experiments

To execute test runs using the included alphabet and hyperparameter values, simply run:
```python
python exps/har/test.py
```

The resuls will be saved to `output/results` directory.

If needed, the alphabet construction and hyperparameter optimization steps can be run as:
```python
python exps/har/build_alphabet.py
python exps/har/optimize_params.py
```

### Electric Motor Condition Monitoring Experiments

To execute test runs using the included alphabet and hyperparameter values, simply run:
```python
python exps/cwru/test.py
```

The resuls will be saved to `output/results` directory.

Likewise, the alphabet construction and hyperparameter optimization steps can be run as:
```python
python exps/cwru/build_alphabet.py
python exps/cwru/optimize_params.py
```

### EEG Eye-State Detection Experiments

The microstate sequence and hyperparameters are included in the repo.
Hence, to execute the test runs, simpy run:
```python
python exps/eeg/test.py
```

Similarly, the preprocessing and hyperparameter optimization steps can be run as:
```python
python exps/eeg/preprocess.py
python exps/eeg/optimize_params.py
```

## License

This project is licensed under the MIT License.
