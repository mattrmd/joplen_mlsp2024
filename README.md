# README

## Installation

```bash
bash -l setup.sh
```

Because of size constraints, the `experiments.json` files are not included. This means that the results are cached but that running `singletask_regclass.ipynb` will run all of the experiments again instead of reading the cached `experiments.json` files. However, the reported values are contained in the `metadata.yaml` files, which are included.

Additionally, the preprocessed data is not included. All of it except for the NanoChem datasets is dynamically downloaded when needed, and the data should only be needed if the entire pipeline is run from scratch.

Rerunning the entire pipeline is projected to take about a week of nonstop computation on a 16 core computer with an RTX 3080 GPU.

## Splitting the data

Run the `process_pmlb.ipynb` notebook for singletask data and the `process_mt_datasets.ipynb` notebook for mulitask data.

## Running the experiments

### Singletask experiments

Run the `singletask_regclass.ipynb` notebook for regression and classification experiments.

### Multitask experiments

Run the `multitask_selection.ipynb` notebook for multitask feature selection experiments.

## Plotting

Run the `paper_plotting.ipynb` notebook to generate the plots for the paper.
