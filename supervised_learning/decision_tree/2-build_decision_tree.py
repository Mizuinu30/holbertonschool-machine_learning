#!/usr/bin/env python3
"""Module that builds a decision tree."""


import numpy as np

class Node:
    """Represents a node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth below the current node."""
        if self.is_leaf:
            return self.depth
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below this node."""
        if only_leaves:
            return self.left_child.count_nodes_below(only_leaves=True) + \
                   self.right_child.count_nodes_below(only_leaves=True)
        return 1 + self.left_child.count_nodes_below() + \
               self.right_child.count_nodes_below()

    def add_prefix(self, text, prefix):
        """Add a prefix to each line in the text."""
        lines = text.split("\n")
        new_text = prefix + lines[0]
        for line in lines[1:]:
            new_text += "\n" + prefix + line
        return new_text

    def __str__(self):
        if self.is_root:
            node_text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            node_text = f"-> node [feature={self.feature}, threshold={self.threshold}]"

        left_child_str = self.add_prefix(str(self.left_child), "    |--")
        right_child_str = self.add_prefix(str(self.right_child), "    |--")
        return f"{node_text}\n{left_child_str}\n{right_child_str}"


class Leaf(Node):
    """Represents a leaf in a decision tree."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth below the current node."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below this node."""
        return 1

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Represents a decision tree."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Calculate the depth of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in the decision tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return str(self.root)
