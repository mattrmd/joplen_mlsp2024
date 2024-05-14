# README

## Installation

```bash
bash -l setup.sh
```

==The zip file for peer review already contains all data and results.==

## Downloading the data

Most of the data will be downloaded automatically when the code is run.
However, some of the data is stored in DVC repositories and needs to be downloaded manually.

```bash
pull_dvc_files() {
    local TARGET_DIRS=("$@")
    for TARGET_DIR in "${TARGET_DIRS[@]}"; do
        find "$TARGET_DIR" -type f -name "*.dvc" | while read -r dvc_file; do
            file_path="${dvc_file%.dvc}"
            dvc pull "$file_path"
        done
    done
}

# pull the NanoChem dataset if you want to run the preprocessing yourself
TARGET_DIRS=()"datasets/general-descriptors-datasets/preprocessed/")
pull_dvc_files "${TARGET_DIRS[@]}"
```

## Downloading cached experiment results

```bash
# Pull the preprocessed datasets
TARGET_DIRS=("datasets/pmlb/processed/class/" "datasets/pmlb/processed/reg/")
pull_dvc_files "${TARGET_DIRS[@]}"

dvc pull "datasets/sarcos/processed/"
dvc pull "datasets/nanoparticle/processed/"

# Pull the parameters
TARGET_DIRS=("experiments/parameters/class/" "experiments/parameters/reg/")
pull_dvc_files "${TARGET_DIRS[@]}"

# Pull the experiment results
TARGET_DIRS=("experiments/ax_runs/class/" "experiments/ax_runs/reg/" "experiments/manual/nanoparticle/" "experiments/manual/sarcos/")
pull_dvc_files "${TARGET_DIRS[@]}"

# Pull the plots
TARGET_DIRS=("experiments/plots/")
pull_dvc_files "${TARGET_DIRS[@]}"
```

## Splitting the data

Run the `process_pmlb.ipynb` notebook for singletask data and the `process_mt_datasets.ipynb` notebook for mulitask data.


## Running the experiments

### Singletask experiments

Run the `singletask_regclass.ipynb` notebook for regression and classification experiments.

### Multitask experiments

Run the `multitask_selection.ipynb` notebook for multitask feature selection experiments.

## Plotting

Run the `paper_plotting.ipynb` notebook to generate the plots for the paper.
