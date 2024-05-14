from __future__ import annotations

from abc import ABC, abstractmethod

import cupy
import numpy as np
from sklearn.metrics import mean_squared_error

from JOPLEn.enums import LossType

from .enums import DTYPE


def sigmoid(x: cupy.ndarray) -> cupy.ndarray:
    return cupy.reciprocal(1 + cupy.exp(-x), dtype=DTYPE)


class Loss(ABC):
    def __init__(self: Loss, loss_type: LossType) -> None:
        self.loss_type = loss_type

    @abstractmethod
    def __call__(
        self: Loss,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
    ) -> float:
        pass

    @abstractmethod
    def grad(
        self: Loss,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        pass

    @abstractmethod
    def predict(
        self: Loss,
        w: cupy.ndarray,
        x: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        pass

    def _raw_output(
        self: Loss,
        w: cupy.ndarray,
        x: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        return cupy.sum((x @ w) * s, axis=1, keepdims=True)

    @abstractmethod
    def error(
        self: Loss,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        pass


class SquaredError(Loss):
    def __init__(self: SquaredError) -> None:
        super().__init__(LossType.regression)

    def __call__(
        self: SquaredError,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
    ) -> float:
        y_pred = self.predict(w, x, s)

        return float(cupy.mean((y_pred - y) ** 2))

    def grad(
        self: SquaredError,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        y_pred = self._raw_output(w, x, s)
        return x.T @ ((y_pred - y) * s) / x.shape[0]

    def predict(
        self: SquaredError,
        w: cupy.ndarray,
        x: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        return self._raw_output(w, x, s)

    def error(
        self: SquaredError,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        return float(mean_squared_error(y_true, y_pred, squared=False))


class LogisticLoss(Loss):
    def __init__(self: LogisticLoss) -> None:
        super().__init__(LossType.binary_classification)

    def encode(self: LogisticLoss, y: np.ndarray) -> np.ndarray:
        return (y * 2) - 1

    def __call__(
        self: LogisticLoss,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
    ) -> float:
        raw_output = self._raw_output(w, x, s)
        y = self.encode(y)

        return float(cupy.mean(cupy.log(1 + cupy.exp(-y * raw_output))))

    def grad(
        self: LogisticLoss,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        raw_output = self._raw_output(w, x, s)
        y = self.encode(y)

        return -x.T @ ((y / (cupy.exp(raw_output * y) + 1)) * s) / x.shape[0]

    def predict(
        self: LogisticLoss,
        w: cupy.ndarray,
        x: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        return sigmoid(self._raw_output(w, x, s))

    def error(
        self: LogisticLoss,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        return np.mean((y_true > 0) == (y_pred > 0))
