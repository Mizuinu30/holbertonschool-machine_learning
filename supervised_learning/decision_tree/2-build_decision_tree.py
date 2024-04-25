#!/usr/bin/env python3
"""Module that builds a decision tree."""


import numpy as np

class Node:
    """
    Represents a node in a decision tree, used for both
    internal decision-making nodes and leaves.

    Attributes:
        feature (int): Index of the feature that this node splits on.
        threshold (float): The value to compare against the feature.
        left_child (Node): Reference to the left child of this node.
        right_child (Node): Reference to the right child of this node.
        is_leaf (bool): True if this node is a leaf, otherwise False.
        is_root (bool): True if this node is the root of the tree.
        depth (int): The depth of the node within the tree.
    """
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
        """
        Returns the maximum depth of the tree rooted at this node.
        
        Returns:
            int: The maximum depth below this node.
        """
        if self.is_leaf:
            return self.depth
        return max(self.left_child.max_depth_below(), self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        Counts all nodes or only leaf nodes below this node.
        
        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: The total number of (leaf) nodes below this node.
        """
        if only_leaves:
            return self.left_child.count_nodes_below(only_leaves=True) + \
                   self.right_child.count_nodes_below(only_leaves=True)
        return 1 + self.left_child.count_nodes_below() + \
               self.right_child.count_nodes_below()

    def add_prefix(self, text, prefix):
        """
        Adds a prefix to each line of the given text.
        Used for creating structured tree visualizations.

        Args:
            text (str): The text to prefix.
            prefix (str): The prefix to add to each line.

        Returns:
            str: The text with the prefix added to each line.
        """
        lines = text.split("\n")
        new_text = prefix + lines[0]
        for line in lines[1:]:
            new_text += "\n" + prefix + line
        return new_text

    def __str__(self):
        """
        String representation of the node for visualization
        purposes, showing the node's parameters.

        Returns:
            str: A string representation of this node.
        """
        if self.is_root:
            node_text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            node_text = f"-> node [feature={self.feature}, threshold={self.threshold}]"

        left_child_str = self.add_prefix(str(self.left_child), "    |--")
        right_child_str = self.add_prefix(str(self.right_child), "    |--")
        return f"{node_text}\n{left_child_str}\n{right_child_str}"


class Leaf(Node):
    """
    Represents a leaf in a decision tree. A leaf does not
    split further and contains a decision value.

    Attributes:
        value (any): The decision value this leaf returns.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Leaf nodes do not have children, so the maximum
        depth below them is their own depth.

        Returns:
            int: The depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts itself as a node. Leaf nodes are 
        always 1 in count as they have no children.

        Returns:
            int: Always 1 for leaf nodes.
        """
        return 1

    def __str__(self):
        """
        Provides a simple string representation of the leaf node,
        showing the decision value.

        Returns:
            str: A string representation of this leaf.
        """
        return f"-> leaf [value={self.value}]"


class DecisionTree:
    """
    Represents the overall decision tree structure capable of
    making predictions based on input features.

    Attributes:
        root (Node): The root node of the decision tree.
        max_depth (int): Maximum depth allowed for the tree.
        min_pop (int): Minimum population required to consider a split at a node.
        seed (int): Seed for the random number generator to ensure reproducibility.
        split_criterion (str): Criterion used to decide how
        nodes are split ('gini', 'entropy', etc.).
        predict (function): Prediction function not implemented in this snippet.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Calculates the depth of the tree from the root node.

        Returns:
            int: The depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes or leaf nodes in the entire tree.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: The total number of (leaf) nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Generates a structured string representation of the
        entire decision tree for visualization.

        Returns:
            str: A string visualization of the decision tree.
        """
        return str(self.root)
