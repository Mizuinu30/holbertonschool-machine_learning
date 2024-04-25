#!/usr/bin/env python3
""" Module that builds a decision tree """

import numpy as np

class Node:
    """ Node of the decision tree """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_root = is_root
        self.depth = depth

    def is_leaf(self):
        """ Determine if this node is a leaf node """
        return not (self.left_child or self.right_child)

    def __str__(self):
        """ Return the string representation of the node """
        if self.is_leaf():
            return (f"-> leaf [value={self.value}]")
        else:
            node_desc = f"-> node [feature={self.feature}, threshold={self.threshold}]\n"
            if self.left_child:
                node_desc += self.left_child_add_prefix(self.left_child.__str__())
            if self.right_child:
                node_desc += self.right_child_add_prefix(self.right_child.__str__())
            return node_desc

    def left_child_add_prefix(self, text):
        """ Add a prefix to the left child """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ Add a prefix to the right child """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

class Leaf(Node):
    """ Leaf node of the decision tree """
    def __init__(self, value, depth=0):
        super().__init__(depth=depth)
        self.value = value

    def __str__(self):
        """ Return the string representation of the leaf """
        return f"-> leaf [value={self.value}] "

class Decision_Tree:
    """ Class that builds a decision tree """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def __str__(self):
        """ Return the string representation of the tree """
        return self.root.__str__()

    def depth(self):
        """ Calculate the depth of the tree """
        return self.root.max_depth_below() if self.root else 0

    def count_nodes(self, only_leaves=False):
        """ Count the number of nodes in the tree """
        return self.root.count_nodes_below(only_leaves=only_leaves) if self.root else 0
