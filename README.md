# README

## Installation

```bash
bash -l setup.sh
```

Because of size constraints, the `experiments.json` files are not included. This means that the results are cached but that running `experiments/singletask_regclass.ipynb` will run all of the experiments again instead of reading the cached `experiments.json` files. However, the reported values are contained in the `metadata.yaml` files, which are included.

Additionally, the preprocessed data is not included. The preprocessed data should only be needed if the entire pipeline is run from scratch. All of the raw data except for the NanoChem datasets is dynamically downloaded when needed and used to generate the preprocessed data.

Rerunning all of our experiments should take approximately 1-2 weeks on a 16 core computer with an RTX 3080 GPU.

## Splitting the data

Run the `experiments/process_pmlb.ipynb` notebook to create train/val/test splits for singletask data and the `experiments/process_mt_datasets.ipynb` notebook for mulitask data.

## Running the experiments

### Singletask experiments

Run the `experiments/singletask_regclass.ipynb` notebook for regression and classification experiments.

### Multitask experiments

Run the `experiments/multitask_selection.ipynb` notebook for multitask feature selection experiments.

## Plotting

Run the `experiments/paper_plotting.ipynb` notebook to generate the plots for the paper.
