/**
 * Remove duplicate values while preserving first-occurrence order.
 */
function uniq(items) {
  return [...new Set(items)].sort();
}

module.exports = { uniq };
