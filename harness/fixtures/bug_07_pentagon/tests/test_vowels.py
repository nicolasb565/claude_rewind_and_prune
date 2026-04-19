import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.counters import count_vowels


def test_count_vowels_lowercase():
    assert count_vowels("hello") == 2
    assert count_vowels("xyz") == 0


def test_count_vowels_ignores_case():
    # Should be case-insensitive: "HELLO" has 2 vowels (E, O), same as "hello"
    assert count_vowels("HELLO") == 2
    assert count_vowels("AEIOU") == 5
