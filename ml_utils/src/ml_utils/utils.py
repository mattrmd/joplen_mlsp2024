import warnings
from collections import Counter, defaultdict
from itertools import combinations
from typing import (
    Dict,
    Generator,
    Iterable,
    List,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import _validate_shuffle_split

# Define Types
StateInt = Union[int, RandomState]


# Define Functions
def comb_iterator(sets: list[Set]) -> Generator[Set, None, None]:
    """Generates every partition of an n-dimensional venn diagram.
    This is necessary because some datasets overlap, and we need
    to find all of the ways that each dataset overlaps so that
    we can make sure that we have no data leakage when performing
    training and testing.

    Args:
        sets (list[Set]): A list of sets, where each set is one\
        component (circle) of the n-dimensional venn-diagram

    Yields:
        Generator[Set,None,None]: A set that holds the elements
        from one partition of the n-dimensional venn diagram
    """

    # Iterate over all possible combinations of the sets
    for i in range(1, len(sets) + 1):
        for comb in combinations(sets, i):
            # Find all sets that are not part of this combination
            comb_comp = list(
                (
                    Counter(map(frozenset, sets)) - Counter(map(frozenset, comb))
                ).elements()
            )

            # set.union doesn't accept an empty list
            if len(comb_comp) > 0:
                union = set.union(*map(set, comb_comp))
            else:
                union = set()

            yield set.intersection(*map(set, comb)) - union


def check_rand_state(
    random_state: Union[int, RandomState],
) -> RandomState:
    # set the random state
    if isinstance(random_state, RandomState):
        return random_state
    elif isinstance(random_state, int):
        return RandomState(random_state)
    else:
        raise ValueError(f"Expected int or RandomState, got {type(random_state)}")


def train_val_test_split(
    datasets: Iterable,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_state: StateInt = 0,
) -> Dict[str, list]:
    """Returns training, validation, and testing datasets using SKLearn's
    train_test_split function. I have rewritten this many times, so this
    implementation should make it more straightforward to use in the future.

    Args:
        datasets (Iterable): The datasets that we want to apply the split to.
        split (Tuple[float,float,float], optional): Tre fraction of train,
        test, and validation data we want in each split.
        Defaults to (0.8,0.1,0.1).
        random_state (StateInt, optional): _description_. Defaults to 0.

    Raises:
        ValueError: All values mult be floats
        ValueError: the first three values must sum to 1 (because we only
        access the first three).

    Returns:
        Dict[str,list]: Training, testing, and validation datasets contained
        in a Dictionary for convenience.
    """

    rng = check_rand_state(random_state)

    # if any parameters are not floats
    if not all(map(lambda x: isinstance(x, float), split)):
        raise ValueError(
            "Must provide values for train, validation, and " "testing fractions"
        )

    if np.around(sum(map(abs, split[:3])), 4) != 1:
        raise ValueError("Sum of positive fractions must add to 1.")

    sum_test_val = sum(split[1:3])

    # Get the train and test+val split
    tmp_datasets = train_test_split(
        *datasets,
        train_size=split[0],
        test_size=sum_test_val,
        shuffle=True,
        random_state=rng,
    )
    # Returned interleaved, so separate
    train_datasets = tmp_datasets[0::2]
    other_datasets = tmp_datasets[1::2]

    # Get the val and test split
    tmp_datasets = train_test_split(
        *other_datasets,
        # Rescale fractions
        train_size=split[1] / sum_test_val,
        test_size=split[2] / sum_test_val,
        shuffle=False,  # already shuffled by the earlier call
        random_state=rng,
    )
    # Returned interleaved, so separate
    test_datasets = tmp_datasets[0::2]
    val_datasets = tmp_datasets[1::2]

    return dict(
        train=train_datasets,
        test=test_datasets,
        val=val_datasets,
    )


def split_df_trteval(
    datasets: Iterable[pd.DataFrame],
    columns: Union[Iterable[str], None] = None,
    label_cols: Union[Iterable[str], None] = None,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_state: StateInt = 0,
    bl_sort: bool = True,
) -> Dict[str, List[pd.DataFrame]]:
    """Takes a list of datasets as dataframes and returns a train/test/validation
    split. This method allows there to be an overlap between datasets without causing
    data leakage (i.e. the same sample will not be in more than one of the training,
    testing, or validation sets with a different label). This method also assumes
    that the index of each dataset is a unique identifier.

    Args:
        datasets (Iterable[pd.DataFrame]): The datasets to be split. This method
        assumes that the dataframe index contains a unique identifier for each
        sample. If there is more than one sample with the same identifier, then
        those samples will never end up in different datasets (incorrect behavior).
        If one sample has multiple identifiers, then the same sample may end up in
        multiple datasets (incorrect behavior).
        columns (Union[Iterable[str], None], optional): The dataframe columns that
        should be used for all datasets. If `None`, then the intersection of all
        columns from `datasets` is used. Defaults to None.
        label_cols (Union[Iterable[str], None], optional): The dataframe columns
        that should be ignored during the intersection because they are labels
        and may be unique to each dataset
        split (Tuple[float,float,float], optional): The ratio of data in the
        train/test/validation splits. Sholud add up to 1.0. Defaults to (0.8,0.1,0.1).
        random_state (StateInt, optional): State used for creating
        bl_sort (bool, optional): Indicates whether the columns are sorted before
        being selected.

    Returns:
        Dict[str,List[pd.DataFrame]]: Dict determines train/test/validation datasets.
        Inner list denotes each dataset in the order provided for `datasets`
    """

    rng = check_rand_state(random_state)

    # extract the indices
    lst_idxs = [set(df.index) for df in datasets]
    lst_cols = [set(df.columns) for df in datasets]

    # If no subset is chosen, we will select all of the ones that are shared
    if columns is None:
        columns = set.intersection(*lst_cols)

    split_idxs: dict[str, list] = defaultdict(list)

    # iterate over all sections of the venn diagram
    for part in comb_iterator(lst_idxs):
        # Skip to avoid error
        if len(part) == 0:
            continue

        part = sorted(part)

        try:
            _validate_shuffle_split(len(part), test_size=split[1], train_size=split[0])
            _validate_shuffle_split(len(part), test_size=split[1], train_size=split[0])

            # Get the train/test/val split for that partition
            dct_splits = train_val_test_split(
                [sorted(list(part))],
                split=split,
                random_state=rng,
            )

            # Add the selected indices to the train/test/val sets
            for k, v in dct_splits.items():
                # Only add the first element because we only sent one
                # list to the split function. Note that we use a list because
                # the elements of any two parts should be disjoint
                split_idxs[k].extend(v[0])
        except ValueError as e:
            # This means that there are not enough samples to properly split
            # Just give all of the samples to the training set
            warnings.warn(
                f"There was an error in generating the dataset. We are ignoring it for now. {str(e)}",
                category=UserWarning,
            )
            # For now, we just add the data to the training dataset
            split_idxs["train"].extend(part)

    # Once all are splits are made, we need to filter each dataset
    out_dict = defaultdict(list)
    columns = sorted(columns) if bl_sort else list(columns)

    # Select the indices from every dataset
    for split_type, lst_idxs in split_idxs.items():
        assert len(lst_idxs) == len(set(lst_idxs)), "There should be no overlap"
        for df in datasets:
            # Get the indices in a stable fashion
            valid_idxs = [idx for idx in lst_idxs if idx in df.index]
            split_df = df.loc[valid_idxs]
            # Select a subset of the columns
            if label_cols is not None:
                ignore_cols = set(split_df.columns).intersection(label_cols)
                ignore_cols = list(ignore_cols - set(valid_idxs))
            else:
                ignore_cols = []
            out_dict[split_type].append(split_df[ignore_cols + columns])

    # return the results of the split
    return dict(out_dict)
