import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collections import dedupe


def test_dedupe_removes_duplicates():
    result = dedupe([3, 1, 2, 1, 3, 2])
    assert set(result) == {1, 2, 3}


def test_dedupe_preserves_order():
    # The first occurrence of each value should be kept, in input order.
    assert dedupe([3, 1, 2, 1, 3]) == [3, 1, 2]
