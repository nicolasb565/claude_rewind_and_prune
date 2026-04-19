import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import average


def test_average_integer_input_exact():
    # When the division is exact, integer input should yield the correct value.
    assert average([2, 4, 6]) == 4


def test_average_returns_float():
    # average([1, 2]) should return 1.5, not 1 (no integer-division truncation).
    assert average([1, 2]) == 1.5
    assert average([1, 2, 3, 4]) == 2.5
