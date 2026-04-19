const { test } = require("node:test");
const assert = require("node:assert");
const { chunk } = require("../src/chunk");

test("chunk preserves all items when length divides evenly", () => {
  const out = chunk([1, 2, 3, 4, 5, 6], 3);
  assert.strictEqual(out.flat().length, 6);
});

test("chunk preserves all items with remainder", () => {
  const out = chunk([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3);
  const total = out.reduce((acc, c) => acc + c.length, 0);
  assert.strictEqual(total, 10, `expected all 10 items preserved, got ${total}`);
});

test("chunk produces roughly equal chunks", () => {
  const out = chunk([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3);
  const sizes = out.map((c) => c.length);
  assert.ok(Math.max(...sizes) - Math.min(...sizes) <= 1, `chunks too uneven: ${sizes}`);
});
