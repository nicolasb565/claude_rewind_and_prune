"""Test suite for the Red-Black Tree. Some tests fail due to a bug."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rbtree import RBTree
import random


def test_insert_and_search():
    tree = RBTree()
    for i in [10, 20, 30, 15, 25, 5, 1]:
        tree.insert(i)
    assert tree.search(15)
    assert tree.search(30)
    assert not tree.search(99)
    ok, msg = tree.verify_properties()
    assert ok, f"After insert: {msg}"
    print("test_insert_and_search: PASS")


def test_delete_simple():
    tree = RBTree()
    for i in [10, 20, 30]:
        tree.insert(i)
    tree.delete(20)
    assert not tree.search(20)
    assert tree.search(10)
    assert tree.search(30)
    ok, msg = tree.verify_properties()
    assert ok, f"After simple delete: {msg}"
    print("test_delete_simple: PASS")


def test_delete_sequence():
    """Delete in a specific order that triggers the fixup bug."""
    tree = RBTree()
    keys = [50, 25, 75, 12, 37, 62, 87, 6, 18, 31, 43]
    for k in keys:
        tree.insert(k)

    # This deletion sequence triggers the mirror case in _delete_fixup
    for k in [75, 87, 62, 50, 43]:
        tree.delete(k)
        ok, msg = tree.verify_properties()
        assert ok, f"After deleting {k}: {msg}"

    assert tree.inorder() == sorted(set(keys) - {75, 87, 62, 50, 43})
    print("test_delete_sequence: PASS")


def test_delete_stress():
    """Randomized stress test — insert 100, delete 70, verify properties."""
    random.seed(0)
    tree = RBTree()
    keys = list(range(100))
    random.shuffle(keys)

    for k in keys:
        tree.insert(k)
    ok, msg = tree.verify_properties()
    assert ok, f"After bulk insert: {msg}"

    to_delete = keys[:70]
    random.shuffle(to_delete)
    for k in to_delete:
        tree.delete(k)
        ok, msg = tree.verify_properties()
        assert ok, f"After deleting {k}: {msg}"

    remaining = sorted(set(keys) - set(to_delete))
    assert tree.inorder() == remaining, f"Inorder mismatch: expected {remaining}, got {tree.inorder()}"
    print("test_delete_stress: PASS")


if __name__ == "__main__":
    test_insert_and_search()
    test_delete_simple()
    test_delete_sequence()
    test_delete_stress()
    print("\nAll tests passed!")
