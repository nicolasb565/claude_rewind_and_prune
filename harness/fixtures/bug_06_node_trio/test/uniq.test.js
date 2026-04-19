const { test } = require("node:test");
const assert = require("node:assert");
const { uniq } = require("../src/uniq");

test("uniq removes duplicates", () => {
  const out = uniq([3, 1, 2, 1, 3, 2]);
  assert.deepStrictEqual([...out].sort(), [1, 2, 3]);
});

test("uniq preserves first-occurrence order", () => {
  assert.deepStrictEqual(uniq([3, 1, 2, 1, 3]), [3, 1, 2]);
});
