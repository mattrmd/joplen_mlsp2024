from enum import Enum, auto

import cupy

DTYPE = cupy.float32


class LossType(Enum):
    multinomial_classification = auto()
    binary_classification = auto()
    regression = auto()
    ranking = auto()


class CellModel(Enum):
    constant = auto()
    linear = auto()


class NormType(Enum):
    """Norm types that are currently supported."""

    L21 = auto()
    LINF1 = auto()
