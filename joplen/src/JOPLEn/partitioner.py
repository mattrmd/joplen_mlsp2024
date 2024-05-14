from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.random import RandomState
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from sklearn.metrics import pairwise_distances_argmin_min
from xgboost import DMatrix, XGBClassifier, XGBRegressor

from .enums import LossType
from .custom_trees import GBLTRegressor, VarMaxForest
from sklearn.linear_model import LinearRegression
from lineartree import LinearForestRegressor, LinearForestClassifier
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


def numpify(x: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    """Convert a pandas DataFrame to a numpy array if necessary.

    Args:
        x (pd.DataFrame | np.ndarray): The input data.

    Returns:
        np.ndarray: The input data as a numpy array.
    """
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.to_numpy()

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    return x


class Partitioner(ABC):
    def __init__(
        self: Partitioner,
        n_cells: int,
        n_partitions: int,
        loss_type: LossType,
        random_state: Union[int, RandomState],
        keep_int: bool = False,
        **model_kwargs: dict[str, Any],
    ) -> None:
        # keep_int is necessary for CatBoost for some reason
        self.n_cells: int = n_cells
        self.n_partitions: int = n_partitions
        self.loss_type: LossType = loss_type
        self.model_kwargs: dict[str, Any] = model_kwargs

        if not isinstance(random_state, RandomState) and not keep_int:
            self.state: RandomState | int = RandomState(random_state)
        else:
            self.state: RandomState | int = random_state

    @abstractmethod
    def partition(self: Partitioner, x: np.ndarray) -> np.ndarray:
        """Take input data and apply a partition mask to it.

        Args:
            self (Partitioner): The partitioner object.
            x (np.ndarray): The input data to be partitioned.

        Returns:
            np.ndarray: The partitioned data.

        """


class VPartition(Partitioner):
    def __init__(
        self: VPartition,
        x: np.ndarray,
        y: np.ndarray,
        n_cells: int,
        n_partitions: int,
        loss_type: LossType,
        random_state: int = 0,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **model_kwargs: dict[str, Any],
    ) -> None:
        """Initializes a VPartition object.

        Args:
            x (np.ndarray): Array of shape (n_samples, n_features) containing the input data.
            y (np.ndarray): Array of shape (n_samples,) containing the target labels.
            n_cells (int): Number of Voronoi cells to create.
            n_partitions (int): Number of partitions to create.
            random_state (int, optional): Seed for the random number generator. Defaults to 0.

        """
        super().__init__(n_cells, n_partitions, loss_type, random_state, **model_kwargs)

        self._create_voronoi(x)

    def _create_voronoi(self: VPartition, x: np.ndarray) -> None:
        idx = self.state.randint(0, x.shape[0], size=(self.n_cells, self.n_partitions))

        self.vor_points = x[idx]

    def partition(self: VPartition, x: np.ndarray) -> np.ndarray:
        """Partition the input data using Voronoi cells.

        Args:
            self (VPartition): The Voronoi partitioner object.
            x (np.ndarray): The input data to be partitioned.

        Returns:
            np.ndarray: The partitioned data wth shape (n_samples, n_partitions)
        """
        # Check if the dimensions are compatible
        if x.shape[1] != self.vor_points.shape[2]:
            msg = (
                "Dimension mismatch between input data and Voronoi points:"
                f"{x.shape[1]}, {self.vor_points.shape[2]}"
            )
            raise ValueError(msg)

        x = numpify(x)

        # Initialize an empty list to store the results
        concatenated_indices = []

        # Iterate through the n different sets of Voronoi points
        for i in range(self.vor_points.shape[1]):
            vor_points_i = self.vor_points[:, i, :]
            indices, _ = pairwise_distances_argmin_min(x, vor_points_i)
            concatenated_indices.append(indices.reshape(-1, 1))

        # Concatenate the results along the specified axis
        return np.hstack(concatenated_indices)

    def plot_partitions(self: VPartition, max_to_plot: int = 1, ax=None) -> None:
        """Plot the Voronoi partitions.

        Args:
            self (VPartition): The Voronoi partitioner object.
            max_to_plot (int, optional): The number of partitions to plot. Defaults to 1.
            ax (_type_, optional): The matplotlib axis to plot on. Defaults to None.

        """
        from matplotlib import cm

        if ax is None:
            fig, ax = plt.subplots()

        colors = cm.get_cmap("tab10").colors

        for i in range(max_to_plot):
            vor = Voronoi(self.vor_points[:, i, :])
            voronoi_plot_2d(
                vor,
                show_vertices=False,
                line_colors=colors[i],
                line_styles="-",
                point_size=0,
                ax=ax,
            )


class TreePartition(Partitioner, ABC):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_cells: int,
        n_partitions: int,
        loss_type: LossType,
        random_state: int = 0,
        prefit_model: Any | None = None,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        keep_int: bool = False,
        **model_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            n_cells,
            n_partitions,
            loss_type,
            random_state,
            keep_int,
            **model_kwargs,
        )

        if prefit_model is None:
            self._fit_model(x, y, x_val, y_val)
        else:
            self._prefit_model(prefit_model, x)

        # should provide all of the leaf indices
        train_leaf_indices = self._get_leaves(x)

        assert len(train_leaf_indices.shape) == 2, "Leaf indices must be 2D"

    def partition(self, x: np.ndarray) -> np.ndarray:
        leaf_indices = self._get_leaves(x)
        # Leaf indices are the partitions
        return leaf_indices.astype(int)

    @abstractmethod
    def _get_leaves(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        pass

    @abstractmethod
    def _prefit_model(self, model: Any, x: np.ndarray) -> None:
        pass


class SKLTreePartition(TreePartition):
    def _truncate_classification(self, leaf_indices: np.ndarray) -> np.ndarray:
        if len(leaf_indices.shape) == 2:
            return leaf_indices

        return leaf_indices[:, :, 0]

    def _get_leaves(self, x: np.ndarray) -> np.ndarray:
        # SKLearn uses the convention that leaf indices start at 1
        leaf_indices = self._truncate_classification(self.model.apply(x))

        # actually rename the leaf indices
        for i in range(leaf_indices.shape[0]):
            for j in range(leaf_indices.shape[1]):
                leaf_indices[i, j] = self.renamer[j][leaf_indices[i, j]]

        return leaf_indices

    def get_leaf_paths(self, x: np.ndarray) -> list[np.ndarray]:
        decision_paths = []

        for tree in self.model.estimators_[:, 0]:
            decision_paths.append(tree.decision_path(x))

        return decision_paths

    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        match self.loss_type:
            case LossType.regression:
                model_class = self.get_regression_model()
            case LossType.binary_classification | LossType.multinomial_classification:
                model_class = self.get_classification_model()
            case _:
                raise ValueError("Loss type not supported")

        self.model = model_class(  # type: ignore[reportOptionalCall]
            n_estimators=self.n_partitions,
            max_leaf_nodes=self.n_cells,
            random_state=self.state,
            **self.model_kwargs,
        )

        self.model.fit(x, y.flatten())

        self._init_leaf_renamer(x)

    def _init_leaf_renamer(self, x: np.ndarray) -> None:
        # Need to use this really hacky method because leaf indices are not
        # sequential

        # Get leaf indices for each data point in each tree
        # SKLearn uses the convention that leaf indices start at 1
        leaf_indices = self._truncate_classification(self.model.apply(x))

        self.renamer = []

        for tree_leaves in leaf_indices.T:
            d = {v: i for i, v in enumerate(np.unique(tree_leaves))}
            self.renamer.append(d)

    def _prefit_model(self, model: Any, x: np.ndarray) -> None:
        self.model = model

        self._init_leaf_renamer(x)

    @abstractmethod
    def get_regression_model(self):
        pass

    @abstractmethod
    def get_classification_model(self):
        pass


class ExtraTreePartition(SKLTreePartition):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_cells: int,
        n_partitions: int,
        loss_type: LossType,
        random_state: int = 0,
        prefit_model: Any | None = None,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **model_kwargs: dict[str, Any],
    ) -> None:
        self.super().__init__(
            x,
            y,
            n_cells,
            n_partitions,
            loss_type,
            random_state,
            prefit_model,
            x_val,
            y_val,
            {**model_kwargs, "max_features": 1},
        )

    def get_regression_model(self):
        return ExtraTreesRegressor

    def get_classification_model(self):
        return ExtraTreesClassifier


class LGBMPartition(TreePartition):
    def _get_leaves(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x, pred_leaf=True)

    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        match self.loss_type:
            case LossType.regression:
                model_class = LGBMRegressor
            case LossType.binary_classification | LossType.multinomial_classification:
                model_class = LGBMClassifier
            case _:
                raise ValueError("Loss type not supported")

        self.model = model_class(
            n_estimators=self.n_partitions,
            num_leaves=self.n_cells,
            random_state=self.state,
            verbose=-1,
        )

        self.model.fit(x, y.flatten(), eval_set=(x_val, y_val.flatten()))

    def _prefit_model(self, model: Any, x: np.ndarray) -> None:
        self.model = model

    def get_leaf_paths(self, x: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError("Leaf paths not implemented for LightGBM")


class CBPartition(TreePartition):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_cells: int,
        n_partitions: int,
        loss_type: LossType,
        random_state: int = 0,
        prefit_model: Any | None = None,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **model_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            x,
            y,
            n_cells,
            n_partitions,
            loss_type,
            random_state,
            prefit_model,
            x_val,
            y_val,
            keep_int=True,
            **model_kwargs,
        )

    def _get_leaves(self, x: np.ndarray) -> np.ndarray:
        return self.model.calc_leaf_indexes(x)

    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        match self.loss_type:
            case LossType.regression:
                model_class = CatBoostRegressor
            case LossType.binary_classification | LossType.multinomial_classification:
                model_class = CatBoostClassifier
            case _:
                raise ValueError("Loss type not supported")

        train_pool = Pool(
            x,
            y.flatten(),
            cat_features=self.model_kwargs["cat_features"],
        )

        val_pool = Pool(x_val, y_val.flatten())

        max_depth = np.log2(self.n_cells)
        assert max_depth % 1 == 0, "n_cells must be a power of 2"

        self.model = model_class(
            max_depth=int(max_depth),
            iterations=self.n_partitions,
            random_state=self.state,
            **self.model_kwargs,
        )

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=False,
        )

        self.n_partitions = self.model.tree_count_

    def _prefit_model(self, model: Any, x: np.ndarray) -> None:
        self.model = model

    def get_leaf_paths(self, x: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError("Leaf paths not implemented for CatBoost")


class RFPartition(SKLTreePartition):
    def get_regression_model(self):
        return RandomForestRegressor

    def get_classification_model(self):
        return RandomForestClassifier


class XGBPartition(TreePartition):
    def _get_leaves(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(DMatrix(x), pred_leaf=True)

    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        warnings.warn(
            "XGBoost partitioner does not work yet. It looks like there isn't "
            "a way to set the exact number of leaves when using "
            "`tree_method='exact'`. For more info, see `max_leaves` in the "
            "XGBoost docs."
        )

        match self.loss_type:
            case LossType.regression:
                model_class = XGBRegressor
            case LossType.binary_classification | LossType.multinomial_classification:
                model_class = XGBClassifier
            case _:
                raise ValueError("Loss type not supported")

        self.model = model_class(
            n_estimators=self.n_partitions,
            max_leaves=self.n_cells,
            random_state=self.state,
            tree_method="exact",
            verbosity=0,
        )

        self.model.fit(x, y.flatten())

        self.booster = self.model.get_booster()

    def _prefit_model(self, model: Any, x: np.ndarray):
        warnings.warn(
            "XGBoost partitioner does not work yet. It looks like there isn't "
            "a way to set the exact number of leaves when using "
            "`tree_method='exact'`. For more info, see `max_leaves` in the "
            "XGBoost docs."
        )

        self.model = model
        self.booster = self.model.get_booster()

    def get_leaf_paths(self, x: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError("Leaf paths not implemented for XGBoost")


class GBPartition(SKLTreePartition):
    def get_regression_model(self):
        return GradientBoostingRegressor

    def get_classification_model(self):
        return GradientBoostingClassifier


class ABPartition(SKLTreePartition):
    def get_regression_model(self):
        return AdaBoostRegressor

    def get_classification_model(self):
        return AdaBoostClassifier


class VarMaxForestPartition(TreePartition):
    def _get_leaves(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_index(x).T

    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        match self.loss_type:
            case LossType.regression:
                model_class = VarMaxForest
            case LossType.binary_classification | LossType.multinomial_classification:
                raise NotImplementedError("Classification not implemented")
            case _:
                raise ValueError("Loss type not supported")

        assert (np.log2(self.n_cells) % 1) == 0, "n_cells must be a power of 2"

        self.model = model_class(
            n_estimators=self.n_partitions,
            max_depth=int(np.log2(self.n_cells)),
            random_state=self.state,
            **self.model_kwargs,
        )

        self.model.fit(x, y.flatten())

    def _prefit_model(self, model: Any, x: np.ndarray):
        self.model = model

    def get_leaf_paths(self, x: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError("Leaf paths not implemented for VarMaxForest")


class LinearForestPartition(SKLTreePartition):
    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        assert (np.log2(self.n_cells) % 1) == 0, "n_cells must be a power of 2"

        if "base_estimator" not in self.model_kwargs:
            self.model_kwargs["base_estimator"] = LinearRegression()

        super()._fit_model(x, y, x_val, y_val)

    def get_regression_model(self):
        return LinearForestRegressor

    def get_classification_model(self):
        return LinearForestClassifier


class LinearBoostPartition(SKLTreePartition):
    def _get_leaves(self, x: np.ndarray) -> np.ndarray:
        partitions = np.empty((x.shape[0], self.n_partitions), dtype=int)

        for i, tree in enumerate(self.model.base_learners):
            partitions[:, i] = tree.apply(x)

        return partitions

    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        assert (np.log2(self.n_cells) % 1) == 0, "n_cells must be a power of 2"

        if "base_estimator" not in self.model_kwargs:
            self.model_kwargs["base_estimator"] = LinearRegression()

        self.super()._fit_model(x, y, x_val, y_val)
