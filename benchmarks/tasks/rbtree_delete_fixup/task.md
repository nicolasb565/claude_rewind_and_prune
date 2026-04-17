You are debugging a Red-Black Tree implementation in Python.

The code is at /scratch (rbtree.py and test_rbtree.py).

The test suite (`python3 test_rbtree.py`) fails on the stress test — after a sequence of deletions, the tree violates the red-black property "red node has only black children".

Insert works correctly. Simple deletes work. The bug only manifests under specific deletion sequences that trigger a particular case in `_delete_fixup`.

Find the bug in rbtree.py and fix it. Run the tests to verify your fix.
