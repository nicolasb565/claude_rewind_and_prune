def count_vowels(s):
    """Return the number of vowels in s. Case-insensitive."""
    return sum(1 for c in s if c in "aeiou")
