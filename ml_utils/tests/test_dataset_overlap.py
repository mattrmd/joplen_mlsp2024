# Set up relative imports
from os.path import abspath, dirname, join
from random import Random, random
from sys import path

# General imports
import pytest
from numpy import dtype

from numpy.random import RandomState
from pandas import DataFrame
from scipy import rand

# region check_rand_state
from ml_utils.utils import check_rand_state

good_states = [0, 1, RandomState(0)]

bad_states = [-1, 0.1, "test", (0,), list(), dict()]


# Checks to make sure that the states are
@pytest.mark.parametrize("rand_state", good_states + bad_states)
def test_check_rand_states(rand_state):
    if rand_state in good_states:
        res = check_rand_state(rand_state)
        assert isinstance(
            res, RandomState
        ), f"Expected RandomState, received {type(res)}"
    else:
        with pytest.raises(ValueError):
            check_rand_state(rand_state)


# endregion

# region train_val_test_split

from sklearn.utils._param_validation import InvalidParameterError

from ml_utils.utils import train_val_test_split

n_points = 100
data = [(i,) * 2 for i in range(n_points)]
columns = [0, 1]
index = list(range(n_points))

single_dataset = DataFrame(columns=columns, data=data, index=index)

lst_good_datasets = [[single_dataset], [single_dataset, single_dataset]]

lst_bad_datasets = [[], [single_dataset, single_dataset.loc[: n_points // 2]]]

lst_good_splits = [
    (0.8, 0.1, 0.1),
    (0.1, 0.2, 0.7),
    (0.1, 0.2, 0.7, 0.0),
    (0.1, 0.2, 0.7, 0.1),
]

lst_bad_splits = [
    (-0.8, 0.1, 0.1),
    (0.1, 0.1, 0.1),
    (0.0, 0.0, 0.0),
]

lst_random_states = list(map(int, RandomState(0).randint(0, 1000, 5)))


@pytest.mark.parametrize(
    "datasets, bad_ds",
    zip(
        lst_good_datasets + lst_bad_datasets,
        [False] * len(lst_good_datasets) + [True] * len(lst_good_datasets),
    ),
)
@pytest.mark.parametrize(
    "splits, bad_split",
    zip(
        lst_good_splits + lst_bad_splits,
        [False] * len(lst_good_splits) + [True] * len(lst_good_splits),
    ),
)
@pytest.mark.parametrize("random_state", lst_random_states)
def test_train_val_test_split(datasets, bad_ds, splits, bad_split, random_state):
    funct = lambda: train_val_test_split(
        datasets,
        split=splits,
        random_state=random_state,
    )

    if bad_ds:
        with pytest.raises(ValueError):
            res = funct()
        return
    elif bad_split:
        if sum(splits) != 1:
            with pytest.raises(ValueError):
                res = funct()
        else:
            with pytest.raises(InvalidParameterError):
                res = funct()
        return
    else:
        res = funct()

    for i, dataset in enumerate(datasets):
        len_res = len(set.union(*[set(v[i].index) for v in res.values()]))
        assert (
            dataset.shape[0] == len_res
        ), f"Expected the same shape, but got {(len_res, dataset.shape[0])}"


# endregion


# region

rng = RandomState(0)

lst_datasets = [
    [
        DataFrame(rng.random((1000, 10)), rng.choice(range(2000), 1000), range(10))
        for _ in range(2)
    ],
]

lst_splits = [(0.8, 0.1, 0.1)]

lst_random_state = [0]


@pytest.mark.parametrize("datasets", lst_datasets)
@pytest.mark.parametrize("splits", lst_splits)
@pytest.mark.parametrize("random_state", lst_random_state)
def test_tvt_stability(datasets, splits, random_state):
    res = []

    for _ in range(2):
        res.append(
            split_df_trteval(
                datasets,
                split=splits,
                random_state=random_state,
            )
        )

    for k in res[0].keys():
        for d1, d2 in zip(res[0][k], res[1][k]):
            assert d1.equals(d2), (d1, d2)


# endregion


# region split_df_trteval

from ml_utils.utils import split_df_trteval

s_per_part = 3
n_parts = 4
gen_index = lambda x: [val + n_points * (i) for i in range(s_per_part) for val in x]
n_points = n_parts * s_per_part
rng = RandomState(0)

lst_datasets = [
    [
        DataFrame(
            columns=[1, 2, 3, 4, 5],
            data=rng.random((n_points, 5)),
            index=gen_index([1, 3, 4, 6]),
        ),
        DataFrame(
            columns=[0, 1, 3, 4, 6],
            data=rng.random((n_points, 5)),
            index=gen_index([1, 2, 3, 5]),
        ),
        DataFrame(
            columns=[0, 1, 2, 3, 7],
            data=rng.random((n_points, 5)),
            index=gen_index([1, 2, 4, 7]),
        ),
    ],
]

lst_label_columns = [None, (5, 6, 7)]

lst_columns = [None, (1, 3)]

lst_splits = [(1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 4, 1 / 4)]

lst_rand_states = [0]


@pytest.mark.parametrize("datasets", lst_datasets)
@pytest.mark.parametrize("columns", lst_columns)
@pytest.mark.parametrize("split", lst_splits)
@pytest.mark.parametrize("label_cols", lst_label_columns)
@pytest.mark.parametrize("rand_state", lst_rand_states)
def test_split_df_trteval(datasets, columns, split, label_cols, rand_state):
    res = split_df_trteval(
        datasets=datasets,
        columns=columns,
        split=split,
        random_state=rand_state,
    )

    if label_cols is None:
        label_cols = [None] * len(datasets)

    for i, (df, label) in enumerate(zip(datasets, label_cols)):
        n_samples = sum([v[i].shape[0] for v in res.values()])
        assert df.shape[0] == n_samples, "Expected these to be the same"
        if label is not None:
            intersect = set(label_cols).intersection(df.columns)
            assert len(intersect) == 1 and label in intersect, intersect


# endregion
