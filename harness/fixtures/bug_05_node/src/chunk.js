/**
 * Split an array into `n` roughly-equal chunks.
 *
 * Every element of `items` must appear in exactly one chunk; the last
 * chunk should receive any remainder so no items are lost.
 */
function chunk(items, n) {
  const k = Math.floor(items.length / n);
  const result = [];
  for (let i = 0; i < n; i++) {
    result.push(items.slice(i * k, (i + 1) * k));
  }
  return result;
}

module.exports = { chunk };
