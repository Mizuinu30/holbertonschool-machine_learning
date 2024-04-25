#!/usr/bin/env python3
""" Module that builds a decision tree """


import numpy as np

class Node:
    """ Node of the decision tree"""
    def __init__(self, depth=0, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Calculate the maximum depth below this node"""
        local_max = self.depth
        if self.left_child:
            local_max = max(local_max, self.left_child.max_depth_below())
        if self.right_child:
            local_max = max(local_max, self.right_child.max_depth_below())
        return local_max

    def count_nodes_below(self, only_leaves=False):
        """ Count the number of nodes below this node"""
        count = 1 if not only_leaves else 0
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves=only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves=only_leaves)
        return count

    def __str__(self):
        """ String representation of this node"""
        result = f"[feature={self.feature}, threshold={self.threshold}, depth={self.depth}]\n"
        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "     " + x + "\n"
        return new_text

class Leaf(Node):
    """ Leaf node of the decision tree"""
    def __init__(self, value, depth=None):
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Calculate the maximum depth below this node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Count the number of nodes below this node"""
        return 1 if only_leaves else 0

    def __str__(self):
        """ String representation of a leaf node """
        return f"-> leaf [value={self.value}] "


class Decision_Tree():
    """ Class that builds a decision tree"""
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """ Calculate the depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Count the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ String representation of the decision tree """
        return str(self.root)
