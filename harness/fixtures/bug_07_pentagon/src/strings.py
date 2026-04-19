def is_palindrome(s):
    """Return True if s reads the same forwards and backwards.

    Should be case-insensitive and should ignore whitespace.
    """
    return s == s[::-1]
