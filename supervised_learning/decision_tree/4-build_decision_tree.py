#!/usr/bin/env python3
""" Module that builds a decision tree """


import numpy as np


class Node:
    """representing a node in a decision tree"""

    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        """calculate the maximum depth below the current node"""
        if self.is_leaf:
            return self.depth
        else:
            return max(self.left_child.max_depth_below(), self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Calculate the number of nodes below this node"""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) + self.right_child.count_nodes_below(only_leaves=True))
        else:
            return (1 + self.left_child.count_nodes_below() + self.right_child.count_nodes_below())

    def update_bounds_below(self):
        """Update the bounds for the node and recursively for its children"""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        # Assuming feature and threshold are set; need to be adjusted based on the actual data handling
        for child in [self.left_child, self.right_child]:
            if child:
                # Adjust bounds based on the current node's feature and threshold
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if self.feature is not None:
                    if child is self.left_child:
                        child.upper[self.feature] = min(child.upper.get(self.feature, np.inf), self.threshold)
                    elif child is self.right_child:
                        child.lower[self.feature] = max(child.lower.get(self.feature, -np.inf), self.threshold)
                child.update_bounds_below()

    def get_leaves_below(self):
        """Get all the leaves below this node."""
        if self.is_leaf:
            return [self]
        else:
            return self.left_child.get_leaves_below() + self.right_child.get_leaves_below()

    def __str__(self):
        """print root or node with feature and threshold then print left and right children"""
        node_text = f"root [feature={self.feature}, threshold={self.threshold}]" if self.is_root else f"-> node [feature={self.feature}, threshold={self.threshold}]"
        left_child_str = self.left_child.__str__() if self.left_child else ""
        right_child_str = self.right_child.__str__() if self.right_child else ""
        return f"{node_text}\n{left_child_str}\n{right_child_str}"


class Leaf(Node):
    """representing a leaf in a decision tree"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        """Leaves do not need to update bounds"""
        pass

    def __str__(self):
        return f"-> leaf [value={self.value}] "


class Decision_Tree:
    """representing a decision tree"""

    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def update_bounds(self):
        """Update bounds for the entire tree starting from the root"""
        self.root.update_bounds_below()

    def depth(self):
        """calculate the depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Calculate the number of nodes in the decision tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """print the root node"""
        return self.root.__str__()

    def get_leaves(self):
        """Get all the leaves in the tree."""
        return self.root.get_leaves_below()
