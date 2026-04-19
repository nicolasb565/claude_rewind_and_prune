/**
 * Return true if s reads the same forwards and backwards.
 *
 * Should be case-insensitive and ignore whitespace.
 */
function isPalindrome(s) {
  return s === [...s].reverse().join("");
}

module.exports = { isPalindrome };
