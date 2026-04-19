# Task

A test in `test/chunk.test.js` is failing: `"chunk preserves all items
with remainder"`. The bug is in `src/chunk.js` — the `chunk` function
loses items when the array length is not evenly divisible by `n`.

Diagnose the bug in `src/chunk.js` and fix it so all tests pass.

Run tests with:
```
node --test test/*.test.js
```

Success: all tests pass.
