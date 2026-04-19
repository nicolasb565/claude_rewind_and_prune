import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ranges import partition


def test_partition_preserves_all():
    items = list(range(10))
    result = partition(items, 3)
    assert sum(len(chunk) for chunk in result) == 10, \
        f"expected all 10 items preserved, got {sum(len(c) for c in result)}"


def test_partition_roughly_equal():
    items = list(range(10))
    result = partition(items, 3)
    # All chunks within 1 element of each other
    sizes = [len(c) for c in result]
    assert max(sizes) - min(sizes) <= 1, f"chunks too uneven: {sizes}"
