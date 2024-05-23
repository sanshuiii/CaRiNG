# LEARNING TEMPORAL CAUSAL REPRESENTATION UNDER NON-INVERTIBLE GENERATION PROCESS
Run the scripts in `scripts/run.sh` to generate results for experiment.

Further details are documented within the code.

### Requirements
To install it, create a conda environment with `Python>=3.7` and follow the instructions below. Note, that the current implementation of CaRiNG requires a GPU.
```
conda create -n caring python=3.7
pip install -e .
```

### Datasets

- NG: `python data_generator/NG.py 12`
- NG-TDMP: `python data_generator/NG-TDMP.py 12`

### Run

- Modify `root_path` to your repo path in `./caring/configs/xxx.yaml`
- Get into `./scrips`
- Execute `run_caring.sh`
