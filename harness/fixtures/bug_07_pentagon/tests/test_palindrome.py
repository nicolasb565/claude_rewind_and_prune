import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strings import is_palindrome


def test_palindrome_simple():
    assert is_palindrome("abcba")
    assert not is_palindrome("abcde")


def test_palindrome_ignores_case_and_spaces():
    assert is_palindrome("Race car")
    assert is_palindrome("A man a plan a canal Panama")
    assert not is_palindrome("hello world")
