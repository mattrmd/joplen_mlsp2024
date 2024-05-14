from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import cupy
import numpy as np

from .enums import DTYPE
from .partitioner import Partitioner, TreePartition
from .proj_l1_ball import euclidean_proj_l1ball


class Penalty(ABC):
    def __init__(
        self: Penalty,
        is_smooth: bool,
    ) -> None:
        self.is_smooth = is_smooth

    def build(
        self: Penalty,
        x_train: np.ndarray,
        x_val: np.ndarray,
        partitioner: Partitioner,
    ) -> None:
        pass

    @abstractmethod
    def __call__(
        self: Penalty,
        w: cupy.ndarray,
        bias_idx: int | None,
        x: cupy.ndarray,
        y_pred: cupy.ndarray,
        s: cupy.ndarray,
        is_train: bool,
    ) -> float:
        pass

    def grad_update(
        self: Penalty,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        x: cupy.ndarray,
        y_pred: cupy.ndarray,
        s: cupy.ndarray,
        is_train: bool,
    ) -> cupy.ndarray:
        raise NotImplementedError(
            "Gradient is not implemented, probably because the penalty is not smooth."
        )

    def prox(
        self: Penalty,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        x: cupy.ndarray,
        y_pred: cupy.ndarray,
        s: cupy.ndarray,
        is_train: bool,
    ) -> cupy.ndarray:
        raise NotImplementedError(
            "Proximal operator is not implemented, probably because the penalty is smooth."
        )


class SquaredFNorm(Penalty):
    def __init__(
        self: SquaredFNorm,
        lam: float = 1.0,
    ) -> None:
        super().__init__(is_smooth=True)

        assert lam > 0, "The regularization parameter must be positive."
        self.lam = lam

    def __call__(
        self: SquaredFNorm,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> float:
        return self.lam * cupy.sum(w[:bias_idx] ** 2) / 2

    def grad_update(
        self: SquaredFNorm,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> cupy.ndarray:
        w[:bias_idx] -= self.lam * mu * w[:bias_idx]
        return w


class NonsmoothGroupPenalty(Penalty, ABC):
    def __init__(
        self: NonsmoothGroupPenalty,
        lam: float = 1.0,
    ) -> None:
        super().__init__(is_smooth=False)

        assert lam > 0, "The regularization parameter must be positive."
        self.lam = lam


class Group21Norm(NonsmoothGroupPenalty):

    def __call__(
        self: Group21Norm,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> float:
        return cupy.linalg.norm(w[:bias_idx], axis=-1, ord=2).sum()

    def prox(
        self: Group21Norm,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> cupy.ndarray:
        norm = cupy.linalg.norm(w[:bias_idx], axis=1, keepdims=True, ord=2)
        w[:bias_idx] *= cupy.maximum(1 - self.lam * mu / norm, 0, dtype=DTYPE)
        return w


# TODO: Should both of these functions be adjusted by the number of partitions?
class GroupInf1Norm(NonsmoothGroupPenalty):

    def __call__(
        self: GroupInf1Norm,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> float:
        return cupy.sum(cupy.max(cupy.abs(w[:bias_idx]), axis=-1))

    def prox(
        self: GroupInf1Norm,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> cupy.ndarray:
        w[:bias_idx] = (
            -self.lam * mu * euclidean_proj_l1ball(w[:bias_idx] / self.lam * mu)
        )

        return w


class NuclearNorm(NonsmoothGroupPenalty):

    def __call__(
        self: NuclearNorm,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> float:
        s = cupy.linalg.svd(w[:bias_idx], full_matrices=True, compute_uv=False)
        return cupy.sum(s)

    def prox(
        self: NuclearNorm,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> cupy.ndarray:
        u, s, v = cupy.linalg.svd(w[:bias_idx], full_matrices=False)
        s = cupy.maximum(s - self.lam * mu, 0, dtype=DTYPE)
        w[:bias_idx] = u @ (s[:, None] * v)
        return w


class L1Norm(Penalty):

    def __init__(
        self: L1Norm,
        lam: float = 1.0,
    ) -> None:
        super().__init__(is_smooth=False)

        assert lam > 0, "The regularization parameter must be positive."
        self.lam = lam

    def __call__(
        self: L1Norm,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> float:
        return self.lam * cupy.sum(cupy.abs(w[:bias_idx]))

    def prox(
        self: L1Norm,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        *_: list,
    ) -> cupy.ndarray:
        s = cupy.sign(w[:bias_idx])
        thresh = cupy.maximum(cupy.abs(w[:bias_idx]) - self.lam * mu, 0, dtype=DTYPE)
        w[:bias_idx] = s * thresh
        return w


class LaplacianType(Enum):
    STANDARD = auto()
    LEFT_NORMALIZED = auto()
    NORMALIZED = auto()


class DistanceWeight(ABC):
    def __init__(
        self: DistanceWeight,
        x_train: np.ndarray,
        x_val: np.ndarray,
        partitioner: Partitioner,
        **params: dict[str, Any],
    ) -> None:
        self.train_weights = self.weight(x_train, partitioner, **params)

        if x_val is not None:
            self.val_weights = self.weight(x_val, partitioner, **params)
        else:
            self.val_weights = None

    def __call__(
        self: DistanceWeight,
        is_train: bool,
    ) -> cupy.ndarray:
        return self.train_weights if is_train else self.val_weights

    @abstractmethod
    def weight(
        self: DistanceWeight,
        x: np.ndarray,
        partitioner: Partitioner,
        **params: dict,
    ) -> np.ndarray:
        raise NotImplementedError("The weight function is not implemented.")


class RBFWeight(DistanceWeight, ABC):
    def __init__(
        self: RBFWeight,
        x_train: np.ndarray,
        x_val: np.ndarray,
        partitioner: Partitioner,
        sigma: float = 1.0,
    ) -> None:
        assert sigma > 0, "Sigma must be positive."

        self.sigma = sigma

        super().__init__(x_train, x_val, partitioner, sigma=sigma)


class EuclidRBFWeight(RBFWeight, ABC):
    def distance(
        self: EuclidRBFWeight,
        x: np.ndarray,
        partitioner: Partitioner,
    ) -> np.ndarray:
        return np.linalg.norm(x[:, None] - x[None, :], axis=-1) ** 2


class TreeRBFWeight(RBFWeight, ABC):
    def __init__(
        self: RBFWeight,
        x_train: cupy.ndarray,
        x_val: cupy.ndarray,
        partitioner: Partitioner,
        sigma: float = 1,
    ) -> None:
        assert issubclass(
            type(partitioner), TreePartition
        ), "The partitioner must be a tree partitioner."

        super().__init__(x_train, x_val, partitioner, sigma)

    def distance(
        self: TreeRBFWeight,
        x: np.ndarray,
        partitioner: Partitioner,
    ) -> np.ndarray:
        leaf_paths = partitioner.get_leaf_paths(x)

        distances = np.empty((len(leaf_paths), x.shape[0], x.shape[0]))

        for i, path in enumerate(leaf_paths):
            path_sum = path.sum(axis=1).A1  # converts to a dense 1D numpy array
            path_dot_path_T = path.dot(path.T).toarray()

            distance = path_sum[:, None] + path_sum[None, :] - 2 * path_dot_path_T
            distances[i] = distance

        return distances.mean(axis=0)


class EuclidGaussWeight(EuclidRBFWeight):
    def weight(
        self: EuclidGaussWeight,
        x: np.ndarray,
        partitioner: Partitioner,
        sigma: float,
    ) -> np.ndarray:
        return np.exp(-self.distance(x, partitioner) * sigma)


class TreeGaussWeight(TreeRBFWeight):
    def weight(
        self: TreeGaussWeight,
        x: np.ndarray,
        partitioner: Partitioner,
        sigma: float,
    ) -> np.ndarray:
        return np.exp(-self.distance(x, partitioner) * sigma)


class EuclidMultiQuadWeight(EuclidRBFWeight):
    def weight(
        self: EuclidMultiQuadWeight,
        x: np.ndarray,
        partitioner: Partitioner,
        sigma: float,
    ) -> np.ndarray:
        weights = self.distance(x, partitioner)
        mask = weights != 0
        weights = np.divide(1, 1 + weights * sigma, where=mask)
        weights[~mask] = 0
        return weights


class TreeMultiQuadWeight(TreeRBFWeight):
    def weight(
        self: TreeMultiQuadWeight,
        x: np.ndarray,
        partitioner: Partitioner,
        sigma: float,
    ) -> np.ndarray:
        weights = self.distance(x, partitioner)
        mask = weights != 0
        weights = np.divide(1, 1 + weights * sigma, where=mask)
        weights[~mask] = 0
        return weights


class Laplacian(Penalty, ABC):
    def __init__(
        self: Laplacian,
        is_smooth: bool,
        lam: float = 1.0,
        sigma: float = 1.0,
        weight_class: type[DistanceWeight] = EuclidGaussWeight,
        laplacian_type: LaplacianType = LaplacianType.STANDARD,
    ) -> None:
        super().__init__(is_smooth=is_smooth)

        assert lam > 0, "The regularization parameter must be positive."
        assert sigma > 0, "The variance parameter must be positive."

        self.lam = lam
        self.sigma = sigma
        self.laplacian_type = laplacian_type
        self.weight_class = weight_class

    def create_laplacian(self: Laplacian, weights: np.ndarray) -> cupy.ndarray:
        if self.laplacian_type == LaplacianType.LEFT_NORMALIZED:
            s = np.sum(weights, axis=1)
            d_inv = np.reciprocal(s, where=s != 0)
            L = d_inv[:, None] * weights
        elif self.laplacian_type == LaplacianType.NORMALIZED:
            s = np.sqrt(np.sum(weights, axis=1))
            d_inv = np.reciprocal(s, where=s != 0)
            L = d_inv[:, None] * weights * d_inv[None, :]
        elif self.laplacian_type == LaplacianType.STANDARD:
            L = weights
        else:
            raise ValueError(f"Invalid Laplacian type: {self.laplacian_type}")

        return cupy.array(L, dtype=DTYPE)

    def build(
        self: Laplacian,
        x_train: np.ndarray,
        x_val: np.ndarray,
        partitioner: Partitioner,
    ) -> None:
        self.weight = self.weight_class(x_train, x_val, partitioner, self.sigma)

        self.L_train = self.create_laplacian(self.weight(True))
        self.L_val = self.create_laplacian(self.weight(False))


class SquaredLaplacian(Laplacian):
    def __init__(
        self: SquaredLaplacian,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            is_smooth=True,
            **kwargs,
        )

    def __call__(
        self: SquaredLaplacian,
        w: cupy.ndarray,
        bias_idx: int | None,
        x: cupy.ndarray,
        y_pred: cupy.ndarray,
        s: cupy.ndarray,
        is_train: bool,
    ) -> float:
        L = self.L_train if is_train else self.L_val

        return self.lam * float(y_pred.T @ L @ y_pred / (2 * x.shape[0]))

    def grad_update(
        self: SquaredLaplacian,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        x: cupy.ndarray,
        y_pred: cupy.ndarray,
        s: cupy.ndarray,
        is_train: bool,
    ) -> cupy.ndarray:
        L = self.L_train if is_train else self.L_val

        # L may not be symmetric
        res = x.T @ ((L @ y_pred) * s)
        res += x.T @ ((L.T @ y_pred) * s)

        return w - self.lam * mu * res / (2 * x.shape[0])


class TVLaplacian(Laplacian):
    def __init__(
        self: TVLaplacian,
        **kwargs: dict,
    ) -> None:
        assert (
            "laplacian_type" not in kwargs
        ), "The Laplacian type is not applicable to the total variation penalty."

        super().__init__(
            is_smooth=False,
            laplacian_type=LaplacianType.STANDARD,
            **kwargs,
        )

    def __call__(
        self: TVLaplacian,
        w: cupy.ndarray,
        bias_idx: int | None,
        x: cupy.ndarray,
        y_pred: cupy.ndarray,
        s: cupy.ndarray,
        is_train: bool,
    ) -> float:
        L = self.L_train if is_train else self.L_val

        return (
            w
            - self.lam * cupy.mean(cupy.abs(y_pred[:, None] - y_pred[None, :]) * L) / 2
        )

    def prox(
        self: TVLaplacian,
        mu: float,
        w: cupy.ndarray,
        bias_idx: int | None,
        x: cupy.ndarray,
        y_pred: cupy.ndarray,
        s: cupy.ndarray,
        is_train: bool,
    ) -> cupy.ndarray:
        raise NotImplementedError("The total variation penalty is not implemented.")
