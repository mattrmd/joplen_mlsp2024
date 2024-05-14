from collections import Counter
from dataclasses import dataclass

import pytest

from ml_utils.utils import comb_iterator

fset = frozenset


@dataclass
class VennTest(object):
    """Class to hold the inputs and expected outputs for the Venn Diagram
    partitions
    """

    diagram: list[set[int]]
    solution: Counter[fset, int]


# 3-d Venn diagram with one element in each partition. Used as building block
# for tests
venn_3d = [fset([1, 3, 4, 6]), fset([1, 2, 3, 5]), fset([1, 2, 4, 7])]

lst_tests = [
    # No circles given
    VennTest(
        diagram=[],
        solution=Counter(),
    ),
    # 1 circle
    VennTest(
        diagram=venn_3d[:1],
        solution=Counter([fset([1, 3, 4, 6])]),
    ),
    # 2 circles with partial overlap
    VennTest(
        diagram=venn_3d[:2],
        solution=Counter([fset([6, 4]), fset([1, 3]), fset([5, 2])]),
    ),
    # 3 circles with all partitions nonempty
    VennTest(
        diagram=venn_3d,
        solution=Counter([fset([i + 1]) for i in range(7)]),
    ),
    # 3 circles with 2 exact overlaps and 1 partial overlap
    VennTest(
        diagram=[venn_3d[0], venn_3d[0], venn_3d[2]],
        solution=Counter(
            {fset([]): 4, fset([4, 1]): 1, fset([2, 7]): 1, fset([6, 3]): 1}
        ),
    ),
    # 2 circles no overlap
    VennTest(
        diagram=[fset([1, 2]), fset([3, 4])],
        solution=Counter([fset([1, 2]), fset([3, 4]), fset()]),
    ),
]


@pytest.mark.parametrize("venn_test", lst_tests)
def test_comb_iterator(venn_test: VennTest):
    # Convert elements to frozen sets so they can be compared as sets
    prediction = Counter(map(fset, comb_iterator(venn_test.diagram)))

    assert (
        prediction == venn_test.solution
    ), f"The sets were not equal: {prediction} {venn_test.solution}"
