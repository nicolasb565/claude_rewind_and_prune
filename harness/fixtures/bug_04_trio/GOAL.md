# Task

Three tests are failing in this repository, each revealing a bug in a
different module:

- `tests/test_dedupe.py::test_dedupe_preserves_order`
- `tests/test_partition.py::test_partition_preserves_all`
- `tests/test_palindrome.py::test_palindrome_ignores_case_and_spaces`

Diagnose and fix all three bugs so all tests pass.

Run tests with:
```
python -m pytest tests/ -x
```

Success: all tests pass.
