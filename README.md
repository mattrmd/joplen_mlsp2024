# README

## Installation

```bash
bash -l setup.sh
```

==The zip file for peer review already contains all data and results.==

Because of size constraints, the `experiments.json` files are not included. This means that the results are cached but that running `singletask_regclass.ipynb` will run all of the experiments again instead of reading the cached `experiments.json` files.

## Splitting the data

Run the `process_pmlb.ipynb` notebook for singletask data and the `process_mt_datasets.ipynb` notebook for mulitask data.

## Running the experiments

### Singletask experiments

Run the `singletask_regclass.ipynb` notebook for regression and classification experiments.

### Multitask experiments

Run the `multitask_selection.ipynb` notebook for multitask feature selection experiments.

## Plotting

Run the `paper_plotting.ipynb` notebook to generate the plots for the paper.
