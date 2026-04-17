#!/bin/bash
# Verify the agent's fix by running the test suite.
# Exits 0 iff all 4 tests pass — directly proxies python3's exit code.
#
# Args: $1 = scratch dir (defaults to /scratch when called by entrypoint)
set -u
SCRATCH="${1:-/scratch}"
cd "$SCRATCH" || exit 2
python3 test_rbtree.py
exit $?
