const { test } = require("node:test");
const assert = require("node:assert");
const { chunk } = require("../src/chunk");

test("chunk preserves all items when length divides evenly", () => {
  const out = chunk([1, 2, 3, 4, 5, 6], 3);
  assert.strictEqual(out.flat().length, 6);
});

test("chunk preserves all items with remainder", () => {
  const out = chunk([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3);
  assert.strictEqual(out.reduce((a, c) => a + c.length, 0), 10);
});
