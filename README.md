# README

## Installation

```bash
bash -l setup.sh
```

Because of size constraints, the `experiments.json` files are not included. This means that the results are cached but the surrogate models used during hyperparameter optimization are not. Running `experiments/singletask_regclass.ipynb` will still skip the experiments with cached `metadata.ymal` files since these contain all information needed for plotting.

Additionally, the raw and preprocessed data are not included for most datasets. The raw data is only included for the NanoChem dataset, which is not easily available elsewhere. The preprocessed data can be computed from the raw data, and should only be needed if the entire pipeline is run from scratch.

Rerunning all of our experiments should take approximately 1-2 weeks on a 16 core computer with an RTX 3080 GPU.

## Splitting the data

Run the `experiments/process_pmlb.ipynb` notebook to create train/val/test splits for singletask data and the `experiments/process_mt_datasets.ipynb` notebook for mulitask data.

## Running the experiments

### Singletask experiments

Run the `experiments/singletask_regclass.ipynb` notebook for regression and classification experiments. You must run `experiments/process_pmlb.ipynb` first to create the train/val/test splits.

### Multitask experiments

Run the `experiments/multitask_selection.ipynb` notebook for multitask feature selection experiments. You must run `experiments/process_mt_datasets.ipynb` first to create the train/val/test splits.

## Plotting

Run the `experiments/paper_plotting.ipynb` notebook to generate the plots for the paper. This will use the cached experiment results unless you delete the contents of `experiments/ax_runs/` and then rerun `experiments/singletask_regclass.ipynb` and `experiments/multitask_selection.ipynb`.
