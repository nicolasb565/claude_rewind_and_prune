def partition(items, n):
    """Split items into n roughly-equal chunks.

    The last chunk should receive any remainder so all items end up somewhere.
    """
    k = len(items) // n
    return [items[i * k:(i + 1) * k] for i in range(n)]
