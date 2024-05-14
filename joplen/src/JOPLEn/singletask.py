from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import cupy
import numpy as np
from numpy.random import RandomState
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from .enums import CellModel, LossType, DTYPE
from .partitioner import Partitioner, numpify
from .st_loss import Loss, SquaredError
from .st_penalty import (
    Laplacian,
    LaplacianType,
    Penalty,
)


class JOPLEn:

    def __init__(
        self: JOPLEn,
        partitioner: type[Partitioner] | Partitioner,
        n_cells: int = 1,
        n_partitions: int = 1,
        random_state: int | RandomState = 0,
        loss_fn: type[Loss] = SquaredError,
        cell_model: CellModel = CellModel.linear,
        x_scaler: StandardScaler | None = None,
        y_scaler: StandardScaler | None = None,
        regularizers: list[Penalty] | None = None,
        part_kwargs: dict[str, float] | None = None,
        mu: float = 0.001,
        max_iters: int = 1000,
        patience: int = 100,
        stop_thresh: float = 1e-6,
        early_stop: bool = True,
        rescale: bool = True,
        mu_decr: float = 0.5,
        use_nesterov: bool = True,
    ) -> None:
        if isinstance(partitioner, type):
            cond = issubclass(partitioner, Partitioner)
        else:
            cond = isinstance(partitioner, Partitioner)

        if not cond:
            raise ValueError("Value of partitioner must be a subclass of Partitioner.")

        if n_cells == 1 and n_partitions > 1:
            raise RuntimeWarning(
                "Multiple partitions with a single cell is redundant and will only increase execution time."
            )

        if n_cells < 1:
            raise ValueError("Number of cells must be greater than 0.")
        if n_partitions < 1:
            raise ValueError("Number of partitions must be greater than 0.")

        if isinstance(partitioner, type):
            self.pclass = partitioner
            self.partitioner: Partitioner | None = None
        else:
            self.pclass = type(partitioner)
            self.partitioner = partitioner

        assert (x_scaler is None) == (
            y_scaler is None
        ), "Either both scalers must be None or neither should be."
        self.x_scaler: StandardScaler | None = x_scaler
        self.y_scaler: StandardScaler | None = y_scaler

        self.n_cells = n_cells
        self.n_partitions = n_partitions
        self.w: cupy.ndarray | None = None
        self.loss_fn: LossType = loss_fn()
        self.bl_regr: bool = self.loss_fn.loss_type == LossType.regression
        self.cell_model: CellModel = cell_model
        self.regularizers = regularizers if regularizers else []

        self.smooth_reg = [r for r in self.regularizers if r.is_smooth]
        self.nonsmooth_reg = [r for r in self.regularizers if not r.is_smooth]
        self.mu = mu
        self.max_iters = max_iters
        self.patience = patience
        self.stop_thresh = stop_thresh
        self.early_stop = early_stop
        self.mu_decr = mu_decr
        self.use_nesterov = use_nesterov
        self.rescale = rescale

        self.part_kwargs: dict[str, float] | None = part_kwargs if part_kwargs else {}

        if not isinstance(random_state, RandomState):
            self.random_state: RandomState = RandomState(random_state)
        else:
            self.random_state: RandomState = random_state

    def _create_partitioner(
        self: JOPLEn,
        x: np.ndarray,
        y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> Partitioner:
        """Create the partitioner for each round.

        Args:
            self (JOPLEn): The JOPLEn object.
            x (np.ndarray): The input data, a numpy array of shape (n_samples,
            n_features).
            y (np.ndarray): The input labels. See x for more details.
        """
        partitioner = self.pclass(
            x,
            y,
            self.n_cells,
            self.n_partitions,
            self.loss_fn.loss_type,
            # TODO: should probably change the way that we handle states
            int(self.random_state.randint(0, 2**32 - 1)),
            x_val=val_x,
            y_val=val_y,
            **self.part_kwargs,
        )

        self.n_partitions = partitioner.n_partitions

        return partitioner

    def _get_cells(
        self: JOPLEn,
        x: np.ndarray,
    ) -> cupy.ndarray:
        """Get the partitions for each round.

        Args:
            self (JOPLEn): The JOPLEn object.
            x (cupy.ndarray): The input data.

        Returns:
            cupy.ndarray: A n_points x n_partitions array of partition indices.
        """
        if self.partitioner is None:
            raise ValueError("Must fit the model before getting partitions.")

        p_idx = self.partitioner.partition(x)

        p_idx = p_idx + np.arange(self.n_partitions) * self.n_cells

        binary_mask = np.zeros((x.shape[0], self.n_partitions * self.n_cells))
        binary_mask[np.arange(x.shape[0])[:, None], p_idx] = 1

        return cupy.asarray(binary_mask, dtype=DTYPE)

    def add_bias(self: JOPLEn, x: np.ndarray) -> tuple[np.ndarray, int | None]:
        """Add a bias term to the input data.

        Args:
            x (np.ndarray): The input data.

        Returns:
            tuple[np.ndarray, int|None]: The input data with the bias term added,
            and an index that can be used to remove the bias term.
        """

        match self.cell_model:
            case CellModel.constant:
                return np.ones((x.shape[0], 1), dtype=DTYPE), None
            case CellModel.linear:
                return np.hstack((x, np.ones((x.shape[0], 1), dtype=DTYPE))), -1
            case _:
                raise ValueError("Invalid cell model.")

    def _score(
        self: JOPLEn,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y_true: cupy.ndarray,
        s: cupy.ndarray,
    ) -> float:
        """Compute the loss function.

        Args:
            self (JOPLEn): The JOPLEn object.
            w (cupy.ndarray): The weight matrix.
            x (cupy.ndarray): The input data.
            y (cupy.ndarray): The input labels.
            s (cupy.ndarray): The partition matrix.

        Returns:
            float: The loss value.
        """
        y_pred = self.loss_fn.predict(w, x, s).get()

        if self.bl_regr and self.y_scaler is not None:
            y_pred = self.y_scaler.inverse_transform(y_pred)
            y_true = self.y_scaler.inverse_transform(y_true.get())
        else:
            y_true = y_true.get()

        return self.loss_fn.error(y_true, y_pred)

    def predict(self: JOPLEn, x: cupy.ndarray) -> np.ndarray:
        """Predict the output for the given input and task.

        Args:
            self (JOPLEn): The JOPLEn object.
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

        x, _ = self.add_bias(x)

        x = cupy.asarray(x, dtype=DTYPE)

        y_pred = self.loss_fn.predict(self.w, x, s).get()

        if self.bl_regr and self.y_scaler is not None:
            return self.y_scaler.inverse_transform(y_pred)
        else:
            return y_pred

    def compute_obj_terms(
        self: JOPLEn,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
        is_train: bool,
    ) -> tuple[dict[str, float], float]:
        objective_terms = {}

        y_pred = self.loss_fn.predict(w, x, s)

        objective_terms["loss"] = self.loss_fn(w, x, y, s)

        for reg in self.regularizers:
            name = reg.__class__.__name__
            objective_terms[name] = float(reg(w, self.bias_idx, x, y_pred, s, is_train))

        return objective_terms, sum(objective_terms.values())

    def compute_obj_grad(
        self: JOPLEn,
        mu: float,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        if len(self.smooth_reg) == 0:
            return w

        y_pred = self.loss_fn.predict(w, x, s)

        w_out = w.copy()

        # apply each smooth regularizer as a gradient update
        for reg in self.smooth_reg:
            w_out = reg.grad_update(mu, w, self.bias_idx, x, y_pred, s, True)

        return w_out

    def compute_obj_prox(
        self: JOPLEn,
        mu: float,
        w: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
    ) -> cupy.ndarray:
        if len(self.nonsmooth_reg) == 0:
            return w

        y_pred = self.loss_fn.predict(w, x, s)

        # Use the proximal average to apply multple proximal operators
        tmp_weights = cupy.zeros_like(w, dtype=DTYPE)

        for reg in self.nonsmooth_reg:
            tmp_weights += reg.prox(mu, w, self.bias_idx, x, y, y_pred, s, True)

        return tmp_weights / len(self.nonsmooth_reg)

    def fit(
        self: JOPLEn,
        x: np.ndarray,
        y: np.ndarray,
        print_epochs: int = 100,
        verbose: bool = True,
        val_x: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Fit the JOPLEn model.

        Args:
            self (JOPLEn): The JOPLEn object.
            x (np.ndarray): The input data, a numpy array of shape (n_samples,
                n_features).
            y (np.ndarray): The input labels. See x for more details.
            core (float, optional): The regularization parameter for the regularization.
            mu (float, optional): The step size. Defaults to 0.001.
            max_iters (int, optional): The maximum number of iterations.
                Defaults to 1000.
            print_epochs (int, optional): The number of epochs between each
                logging. Defaults to 100.
            verbose (bool, optional): Whether to print the logs. Defaults to
                True.
            val_x (Union[np.ndarray, None], optional): The validation
                input data. Defaults to None.
            val_y (Union[np.ndarray, None], optional): The validation
                input labels. Defaults to None.
            feat_thresh (float, optional): The feat_thresh for a feature to be
                considered selected. Defaults to 1e-3.
            core_prox (Callable[[cupy.ndarray, float], cupy.ndarray], optional):
                The proximal operator for the weights. Defaults to l21_prox.
        """
        # Rescale/preprocess the data
        x = numpify(x)
        y = numpify(y)

        mu = self.mu

        if self.rescale:
            if self.x_scaler is None:
                self.x_scaler = StandardScaler().fit(x)
                if self.bl_regr:
                    self.y_scaler = StandardScaler().fit(y)

            x = self.x_scaler.transform(x)
            if self.bl_regr:
                y = self.y_scaler.transform(y)

        # fit the partitioner and cache the partitions
        # This check allows us to set the partitioner externally for experiments
        if self.partitioner is None:
            self.partitioner = self._create_partitioner(x, y, val_x, val_y)

        for reg in self.regularizers:
            reg.build(x, val_x, self.partitioner)

        s = self._get_cells(x)

        x, self.bias_idx = self.add_bias(x)

        # Move the training data to the GPU
        x = cupy.asarray(x, dtype=DTYPE)
        y = cupy.asarray(y, dtype=DTYPE)

        using_val = val_x is not None and val_y is not None
        if using_val:
            val_x = numpify(val_x)
            val_y = numpify(val_y)

            if self.rescale:
                val_x = self.x_scaler.transform(val_x)
                if self.bl_regr:
                    val_y = self.y_scaler.transform(val_y)

            val_s = self._get_cells(val_x)

            val_x, _ = self.add_bias(val_x)

            val_x = cupy.asarray(val_x, dtype=DTYPE)
            val_y = cupy.asarray(val_y, dtype=DTYPE)
        else:
            val_x = None
            val_y = None
            val_s = None

        w_prev = cupy.zeros(
            (
                x.shape[1],
                self.n_partitions * self.n_cells,
            ),
            dtype=DTYPE,
        )
        w_next = w_prev
        m_next = w_prev

        history = defaultdict(lambda: defaultdict(list))

        best_val = np.inf
        best_val_w = w_prev
        best_val_idx = 0
        obj_prev = self.loss_fn(w_next, x, y, s)

        best_test_w = w_prev

        t_curr = 1
        t_next = 1
        # proximal gradient descent
        for i in range(self.max_iters):
            w_tmp = w_next
            # Perform accelerated gradient descent
            # EECS 559, Lecture 8, APGD (Nesterov Momentum)
            if self.use_nesterov:
                t_next = (1 + cupy.sqrt(1 + 4 * t_curr**2, dtype=DTYPE)) / 2
                beta = (t_curr - 1) / t_next
                m_next = w_next + beta * (w_next - w_prev)

                # loss gradient
                w_next = m_next - mu * self.loss_fn.grad(m_next, x, y, s)
            else:
                w_next = w_prev - mu * self.loss_fn.grad(w_prev, x, y, s)

            # apply each smooth regularizer as a gradient update
            w_next = self.compute_obj_grad(mu, w_next, x, y, s)
            w_next = self.compute_obj_prox(mu, w_next, x, y, s)

            # Compute the objective function
            obj_terms, obj_next = self.compute_obj_terms(w_next, x, y, s, True)

            w_prev = w_tmp

            # logging
            if (i + 1) % print_epochs == 0:
                res = self.record_performance(
                    w_next,
                    x,
                    y,
                    s,
                    i,
                    verbose,
                    val_x,
                    val_y,
                    val_s,
                )

                for k, v in res.items():
                    for kk, vv in v.items():
                        history[k][kk].append(vv)

            if self.early_stop and using_val and i > 0:
                _, val_obj = self.compute_obj_terms(w_next, val_x, val_y, val_s, False)

                if best_val > val_obj:
                    best_val = val_obj
                    best_val_w = w_next
                    best_val_idx = i
                elif (i - best_val_idx > self.patience) and (
                    best_val - val_obj < self.stop_thresh
                ):
                    break
            else:
                best_val_w = w_next
                best_val_idx = i

            if obj_next > obj_prev:
                # if t_next == 1 and t_next == 1:
                mu *= self.mu_decr

                w_next = best_test_w
                w_prev = best_test_w
                t_next = 1
                t_curr = 1
            else:
                best_test_w = w_next
                t_curr = t_next

            if cupy.abs(obj_next - obj_prev) / cupy.abs(obj_prev) <= self.stop_thresh:
                break

        self.w = best_val_w

        return dict({k: dict(v) for k, v in history.items()})

    def record_performance(
        self: JOPLEn,
        w_next: cupy.ndarray,
        x: cupy.ndarray,
        y: cupy.ndarray,
        s: cupy.ndarray,
        i: int,
        verbose: bool,
        val_x: cupy.ndarray | None,
        val_y: cupy.ndarray | None,
        val_s: cupy.ndarray | None,
    ) -> Any:
        current_time = datetime.now().strftime("%H:%M:%S")
        report_str = f"[{current_time}]:"

        train_loss = self._score(w_next, x, y, s)
        report_str += f" Epoch {i + 1:>6d} | TrL: {train_loss:.6f}"

        if val_x is not None and val_y is not None:
            val_loss = self._score(
                w_next,
                val_x,
                val_y,
                val_s,
            )
            report_str += f" | VaL: {val_loss:.6f}"
        else:
            val_loss = None

        output = {
            "train": {
                "loss": train_loss,
            },
            "val": {
                "loss": val_loss,
            },
        }

        train_obj_terms, train_obj = self.compute_obj_terms(w_next, x, y, s, True)

        output["train"].update(train_obj_terms)

        if val_x is not None and val_y is not None:
            val_obj_terms, val_obj = self.compute_obj_terms(
                w_next, val_x, val_y, val_s, False
            )
            output["val"].update(val_obj_terms)

        # Compensate since all terms are scaled by P for the gradient
        report_str += f" | Obj: {train_obj:.6f}"

        if verbose:
            print(report_str)

        return output

    def get_weights(self: JOPLEn) -> np.ndarray:
        """Get the weights of the model.

        Returns:
            np.ndarray: The weights of the model.
        """
        if self.w is None:
            raise NotFittedError("Must fit the model before getting weights.")

        return self.w.get()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    from .partitioner import RFPartition

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

    jp = JOPLEn(
        partitioner=RFPartition,
        n_cells=20,
        n_partitions=1000,
        loss_fn=SquaredError,
        cell_model=CellModel.constant,
        mu=1e-3,
        max_iters=1000,
        early_stop=True,
        rescale=False,
        regularizers=[
            # SquaredFNorm(lam=0.01),
            # L1Norm(lam=0.01),
            Laplacian(lam=100, laplacian_type=LaplacianType.LEFT_NORMALIZED),
            # Laplacian(lam=1.0, laplacian_type=LaplacianType.NORMALIZED),
            # Group21Norm(lam=0.01),
            # GroupInf1Norm(lam=0.01),
        ],
    )

    history = jp.fit(
        x_train,
        y_train,
        val_x=x_val,
        val_y=y_val,
        print_epochs=10,
    )

    y_pred = jp.predict(x_test)

    print("JOPLEn MSE:", mean_squared_error(y_test, y_pred, squared=False))

    print(
        "Naive MSE:",
        mean_squared_error(
            y_test, y_train.mean() * np.ones_like(y_test), squared=False
        ),
    )

    n_terms = len(history["train"].keys())
    fig, axs = plt.subplots(1, n_terms, figsize=(5 * n_terms, 5))

    if n_terms == 1:
        axs = [axs]

    for ax, (k, v) in zip(axs, history["train"].items()):
        ax.plot(v)
        ax.set_title(k)

    # print(history)

    plt.show()
