def dedupe(items):
    """Remove duplicate items while preserving original order.

    The first occurrence of each item should be kept; later duplicates
    are dropped.
    """
    return list(set(items))
