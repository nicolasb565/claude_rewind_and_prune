# Task

Three tests are failing in this repository, each revealing a bug in a
different module:

- `test/chunk.test.js` — `chunk preserves all items with remainder`
- `test/uniq.test.js` — `uniq preserves first-occurrence order`
- `test/palindrome.test.js` — `isPalindrome ignores case and spaces`

Diagnose and fix all three bugs so all tests pass.

Run tests with:
```
node --test test/*.test.js
```

Success: all tests pass.
