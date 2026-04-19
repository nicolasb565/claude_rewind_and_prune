# Task

The red-black tree implementation in `src/rbtree.py` has a bug in the
deletion fix-up logic. Tests in `tests/test_rbtree.py` exercise insert,
simple delete, a specific deletion sequence, and a randomized stress
pass. The simpler tests pass; `test_delete_sequence` and
`test_delete_stress` fail because the tree's invariants break after
certain deletions.

Diagnose the bug in `src/rbtree.py::RBTree._delete_fixup` (or whichever
helper is responsible) and fix it so all tests pass.

Run tests with:
```
python -m pytest tests/ -x
```

Success: all tests pass.
