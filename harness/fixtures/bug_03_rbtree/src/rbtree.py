"""
Red-Black Tree implementation.

Properties that must hold:
1. Every node is red or black
2. Root is black
3. Every NIL leaf is black
4. Red node has only black children
5. All paths from a node to its descendant NILs have the same black count
"""

RED = True
BLACK = False


class Node:
    def __init__(self, key, color=RED, left=None, right=None, parent=None):
        self.key = key
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

    def __repr__(self):
        c = "R" if self.color == RED else "B"
        return f"Node({self.key}, {c})"


class RBTree:
    def __init__(self):
        self.NIL = Node(key=None, color=BLACK)
        self.root = self.NIL

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, y):
        x = y.left
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x

    def insert(self, key):
        node = Node(key, color=RED, left=self.NIL, right=self.NIL)
        y = None
        x = self.root
        while x != self.NIL:
            y = x
            if node.key < x.key:
                x = x.left
            else:
                x = x.right
        node.parent = y
        if y is None:
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node
        self._insert_fixup(node)

    def _insert_fixup(self, z):
        while z.parent and z.parent.color == RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right  # uncle
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left  # uncle
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(z.parent.parent)
        self.root.color = BLACK

    def _transplant(self, u, v):
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, node):
        while node.left != self.NIL:
            node = node.left
        return node

    def delete(self, key):
        z = self._search(self.root, key)
        if z == self.NIL:
            return False

        y = z
        y_original_color = y.color

        if z.left == self.NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right == self.NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        if y_original_color == BLACK:
            self._delete_fixup(x)
        return True

    def _delete_fixup(self, x):
        while x != self.root and x.color == BLACK:
            if x == x.parent.left:
                w = x.parent.right  # sibling
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left  # sibling
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        x.color = BLACK

    def _search(self, node, key):
        if node == self.NIL or key == node.key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)

    def search(self, key):
        result = self._search(self.root, key)
        return result != self.NIL

    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node != self.NIL:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)

    def verify_properties(self):
        """Verify all 5 RB tree properties. Returns (ok, message)."""
        if self.root == self.NIL:
            return True, "Empty tree"

        # Property 2: root is black
        if self.root.color != BLACK:
            return False, "Root is not black"

        # Property 4: red nodes have black children
        def check_red(node):
            if node == self.NIL:
                return True
            if node.color == RED:
                if node.left.color == RED or node.right.color == RED:
                    return False
            return check_red(node.left) and check_red(node.right)

        if not check_red(self.root):
            return False, "Red node has red child"

        # Property 5: all paths have same black count
        def black_height(node):
            if node == self.NIL:
                return 1
            left_bh = black_height(node.left)
            right_bh = black_height(node.right)
            if left_bh == -1 or right_bh == -1 or left_bh != right_bh:
                return -1
            return left_bh + (1 if node.color == BLACK else 0)

        bh = black_height(self.root)
        if bh == -1:
            return False, "Black height mismatch"

        return True, f"OK (black height={bh})"
