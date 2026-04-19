#!/usr/bin/env bash
# Pass if node's built-in test runner reports all tests passing.
# Exits 0 on success, non-zero on any failure.
cd "$(dirname "$0")"
node --test test/*.test.js
