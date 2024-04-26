#!/usr/bin/env python3
""" Module providing a decision tree implementation for classification and regression tasks. """

import numpy as np

class Node:
    """ A decision tree node. """
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        """ Initializes a tree node with optional feature, threshold, children, and depth attributes. """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth

    def max_depth_below(self):
        """ Returns the maximum depth below this node. """
        if self.is_leaf:
            return self.depth
        return max(self.left_child.max_depth_below(), self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """ Counts all nodes or only leaf nodes below this node. """
        if self.is_leaf:
            return 1
        if only_leaves:
            return self.left_child.count_nodes_below(True) + self.right_child.count_nodes_below(True)
        return 1 + self.left_child.count_nodes_below() + self.right_child.count_nodes_below()

    def left_child_add_prefix(self, text):
        """ Prefixes text visualization for the left child. """
        lines = text.split("\n")
        return "    +--" + lines[0] + "\n" + "\n".join("    |  " + line for line in lines[1:])

    def right_child_add_prefix(self, text):
        """ Prefixes text visualization for the right child. """
        lines = text.split("\n")
        return "    +--" + lines[0] + "\n" + "\n".join("      " + line for line in lines[1:])

    def __str__(self):
        """ String representation of this node. """
        node_text = "root [feature={}, threshold={}]".format(self.feature, self.threshold) if self.is_root else "-> node [feature={}, threshold={}]".format(self.feature, self.threshold)
        return f"{node_text}\n{self.left_child_add_prefix(str(self.left_child))}{self.right_child_add_prefix(str(self.right_child))}"

class Leaf(Node):
    """ A leaf node in a decision tree. """
    def __init__(self, value, depth=0):
        """ Initializes a leaf with a fixed value and depth. """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Returns the depth of this leaf. """
        return self.depth

    def __str__(self):
        """ String representation of this leaf. """
        return f"-> leaf [value={self.value}]"

class Decision_Tree:
    """ A decision tree for data classification or regression. """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        """ Initializes a decision tree with specified parameters. """
        self.rng = np.random.default_rng(seed)
        self.root = root or Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """ Returns the depth of the tree. """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Counts all nodes or only leaves in the tree. """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ String representation of the decision tree. """
        return str(self.root)
