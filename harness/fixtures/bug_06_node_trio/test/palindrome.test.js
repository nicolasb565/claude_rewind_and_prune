const { test } = require("node:test");
const assert = require("node:assert");
const { isPalindrome } = require("../src/palindrome");

test("isPalindrome simple", () => {
  assert.ok(isPalindrome("abcba"));
  assert.ok(!isPalindrome("abcde"));
});

test("isPalindrome ignores case and spaces", () => {
  assert.ok(isPalindrome("Race car"), "'Race car' should be a palindrome");
  assert.ok(isPalindrome("A man a plan a canal Panama"));
  assert.ok(!isPalindrome("hello world"));
});
