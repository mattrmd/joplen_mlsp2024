# Allows a class to reference itself during a type hint
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DataSize = dict[str, tuple[int, int, int]] | None


@dataclass
class Loss:
    """Class for holding Loss scores."""

    rmse: float
    raw_rmse: float
    exp_var: float
    r2: float


@dataclass
class Metrics:
    """Class for holding prediction metrics"""

    loss: dict[str, Loss]
    n_u_feats: int
    n_t_feats: dict[str, int]
    ds_size: DataSize = None


@dataclass
class SelFeats:
    """Class for holding selected features."""

    u_feats: tuple[str]
    t_feats: dict[str, tuple[str]]
    n_total_feats: int
    metadata: Optional[dict[str, Any]] = None


@dataclass
class MetaData:
    _start_time: Optional[float] = None
    _end_time: Optional[float] = None
    runtime: Optional[float] = None

    def start_experiment(self):
        self._start_time = time.time()

    def end_experiment(self):
        self._end_time = time.time()

        if self._start_time and self._end_time:
            runtime: float = self._end_time - self._start_time
            if runtime > 0:
                self.runtime = runtime
            else:
                raise ValueError("The start time must be before the end time")
        else:
            raise ValueError(
                "You must record both the start and the end time before computing the runtime"
            )


@dataclass
class MetricWrapper:
    """Class for holding metrics/outputs from model training"""

    metrics: Metrics
    params: dict
    sel_feats: SelFeats
    metadata: Optional[dict[str, Any]] = None
