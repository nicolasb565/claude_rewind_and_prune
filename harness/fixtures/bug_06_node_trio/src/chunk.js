function chunk(items, n) {
  const k = Math.floor(items.length / n);
  const result = [];
  for (let i = 0; i < n; i++) {
    result.push(items.slice(i * k, (i + 1) * k));
  }
  return result;
}

module.exports = { chunk };
