from __future__ import annotations
from collections import defaultdict
from datetime import datetime

from sklearn.exceptions import NotFittedError

import numpy as np
from .enums import LossType, NormType
import cupy
from sklearn.preprocessing import StandardScaler
from .enums import DTYPE, numpify
import warnings
from numpy.random import RandomState
from .partitioner import Partitioner


class Booster:
    def __init__(
        self: Booster,
        partitioner: type[Partitioner],
        n_cells: int = 1,
        n_partitions: int = 1,
        random_state: int | RandomState = 0,
    ) -> None:
        if not issubclass(partitioner, Partitioner):
            msg = f"Expected partitioner to be a subclass of Partitioner, got {type(partitioner)}"
            raise TypeError(msg)

        self.pclass = partitioner
        self.partitioner: Partitioner | None = None
        self.n_cells = n_cells
        self.n_partitions = n_partitions

        if not isinstance(random_state, RandomState):
            self.random_state: RandomState = RandomState(random_state)
        else:
            self.random_state: RandomState = random_state

        self.w: cupy.ndarray | None = None
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: StandardScaler | None = None

    def _create_partitioner(
        self: Booster,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Partitioner:
        """Create the partitioner for each round.

        Args:
            self (Booster): The Booster object.
            x (np.ndarray): The input data, a numpy array of shape (n_samples,
            n_features).
            y (np.ndarray): The input labels. See x for more details.
        """
        return self.pclass(
            x,
            y,
            self.n_cells,
            self.n_partitions,
            LossType.regression,
            # TODO: should probably change the way that we handle states
            self.random_state.randint(0, 2**32 - 1),
        )

    def _get_cells(
        self: Booster,
        x: np.ndarray,
    ) -> cupy.ndarray:
        """Get the partitions for each round.

        Args:
            self (Booster): The Booster object.
            x (cupy.ndarray): The input data.

        Returns:
            cupy.ndarray: A n_points x n_partitions array of partition indices.
        """
        if self.partitioner is None:
            raise ValueError("Must fit the model before getting partitions.")

        p_idx = self.partitioner.partition(x)[:, : self.n_partitions]
        p_idx = p_idx + np.arange(self.n_partitions) * self.n_cells

        binary_mask = np.zeros((x.shape[0], self.n_partitions * self.n_cells))
        binary_mask[np.arange(x.shape[0])[:, None], p_idx] = 1

        return cupy.asarray(binary_mask, dtype=DTYPE)

    def predict(self: Booster, x: cupy.ndarray) -> np.ndarray:
        """Predict the output for the given input and task.

        Args:
            self (Booster): The Booster object.
            x (cupy.ndarray): The input data.

        Returns:
            cupy.ndarray: The predicted output.
        """
        if self.w is None:
            raise NotFittedError("Must fit the model before predicting.")

        x = numpify(x)

        if self.x_scaler is not None:
            x = self.x_scaler.transform(x)
        s = self._get_cells(x)

        x = np.ones((x.shape[0], 1), dtype=DTYPE)

        x = cupy.asarray(x)

        y_pred = np.sum(x @ self.w * s, axis=1, keepdims=True).get()

        return self.y_scaler.inverse_transform(y_pred)

    def fit(
        self: Booster,
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        val_x: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        x = numpify(x)
        y = numpify(y)

        self.x_scaler = StandardScaler().fit(x)
        self.y_scaler = StandardScaler().fit(y)

        x = self.x_scaler.transform(x)
        y = self.y_scaler.transform(y)

        self.partitioner = self._create_partitioner(x, y)

        s = self._get_cells(x)

        # Move the training data to the GPU
        x = cupy.ones((x.shape[0], 1), dtype=DTYPE)
        y = cupy.asarray(y, dtype=DTYPE)

        using_val = val_x is not None and val_y is not None
        if using_val:
            val_x = numpify(val_x)
            val_y = numpify(val_y)

            val_x = self.x_scaler.transform(val_x)
            val_y = self.y_scaler.transform(val_y)

            val_s = self._get_cells(val_x)

            val_x = cupy.ones((val_x.shape[0], 1), dtype=DTYPE)

            val_y = cupy.asarray(val_y, dtype=DTYPE)

        else:
            val_x = None
            val_y = None
            val_s = None

        history = defaultdict(list)

        best_val = np.inf
        best_idx = -1

        w = cupy.zeros((1, self.n_partitions * self.n_cells), dtype=DTYPE)

        for p in range(self.n_partitions):
            learn_slice = slice(p * self.n_cells, (p + 1) * self.n_cells)
            eval_slice = slice(0, (p + 1) * self.n_cells)

            if p > 0:
                residuals = y - np.sum(
                    x @ w[:, eval_slice] * s[:, eval_slice], axis=1, keepdims=True
                )
            else:
                residuals = y

            history["train_loss"].append(float(np.mean(residuals**2)))

            current_time = datetime.now().strftime("%H:%M:%S")
            report_str = f"[{current_time}]: "
            report_str += f"{p + 1}/{self.n_partitions} | "
            report_str += f"Tr: {history['train_loss'][-1]:.4f} | "

            w[:, learn_slice] = np.mean(residuals * s[:, learn_slice], axis=0)[None, :]

            if using_val and p > 0:
                val_residuals = val_y - np.sum(
                    val_x @ w[:, eval_slice] * val_s[:, eval_slice],
                    axis=1,
                    keepdims=True,
                )

                val_loss = float(np.mean(val_residuals**2))
                history["val_loss"].append(val_loss)
                report_str += f"Val: {val_loss:.4f} | "

                if val_loss < best_val:
                    best_val = val_loss
                    best_idx = p

            if verbose:
                print(report_str)

        self.w = w[:, : (best_idx + 1) * self.n_cells]
        self.n_partitions = best_idx + 1

        return dict(history)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    from .partitioner import VPartition
    from .singletask import SquaredError
    from .enums import CellModel

    # Parameters for make_regression
    n_samples = 1000  # Adjust this to your dataset size
    n_features = 20  # Adjust this to the number of features you want
    n_informative = 10  # Adjust this to the number of informative features
    n_targets = 1  # Number of targets (output)

    # Generate regression data
    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=0.0,
        random_state=0,
    )

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    print(x.shape)

    # Further split the test set into validation and test sets
    x_val, x_test, y_val, y_test = train_test_split(
        x_test,
        y_test,
        test_size=0.5,
        random_state=0,  # Adjust the test_size to get 25% of the original for validation
    )

    jp = Booster(
        partitioner=VPartition,
        n_cells=20,
        n_partitions=1000,
    )

    history = jp.fit(
        x_train,
        y_train,
        val_x=x_val,
        val_y=y_val,
    )

    y_pred = jp.predict(x_test)
    print("Booster MSE:", mean_squared_error(y_test, y_pred, squared=False))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(history["train_loss"])
    axs[1].plot(history["val_loss"])

    axs[0].set_title("Training Loss")
    axs[1].set_title("Validation Loss")

    plt.show()
