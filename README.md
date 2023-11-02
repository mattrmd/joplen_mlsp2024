# README

## Technical Details

```bash
conda create --prefix ./my_env python=3.10
conda activate ./my_env
conda config --set env_prompt '(my_env) '

pip install -r requirements.txt
```

## Experiments

==Currently focusing on datasets with all scaler values because JOPLEn was not designed to work with categorical, binary, nominal, or ordinal values.==

### Single-task prediction

#### Raw JOPLEn

Methods to evaluate

- Constant cells
  - [ ] SKLearn Gradient Boosting
  - [ ] SKLearn Random Forest
  - [ ] SKLearn Extra Trees (random)
  - [ ] JOPLEn Voronoi cell
- Linear cells
  - [ ] Linear Gradient Boosted Trees
  - [ ] Linear Random Forest
  - [ ] JOPLEn Voronoi cell

##### Notes

- Seems like JOPLEn loses to all but Extra Trees. Also, performs worse than predicting the mean. Seems weird. Need to look into that more. Most ensembles chose the max number of partitions. Might mean that I need to add more partitions to make up for the randomness in the partitions.

#### JOPLEn refitting tree partitions

- Constant cells
  - [ ] JOPLEn with SKLearn Gradient Boosting partitions
  - [ ] JOPLEn with SKLearn Random Forest
  - [ ] JOPLEn with SKLearn Extra Trees (random)
- Linear cells
  - [ ] JOPLEn with Linear Gradient Boosted Trees

#### Datasets

- [Penn Machine Learning Benchmark](https://epistasislab.github.io/pmlb/)

### Single-task feature selection

#### Models

- [ ] ElasticNet
- [ ] Gradient Boosting feature selection
- [ ] JOPLEn with linear partitions

#### Datasets

-

### Multitask feature selection

#### Models

- [ ] Dirty LASSO
- [ ] JOPLEn with linear partitions

#### Datasets

-

## Competing methods

- [Linear Gradient Boosted Trees](https://github.com/cerlymarco/linear-tree)
- [SKLearn Gradient Boosted Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [SKLearn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [SKLearn Extra Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [Dirty LASSO](https://hichamjanati.github.io/mutar/generated/mutar.DirtyModel.html#mutar.DirtyModel)
- [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [Kernel LASSO](https://github.com/riken-aip/pyHSICLasso/)

## Feature Ideas

- [ ] $$\ell_1$$ group penalty across all weights in a partition rather than all cells. This would allow the model to ``learn'' the partitions via sparsity.
