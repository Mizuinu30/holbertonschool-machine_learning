#!/usr/bin/env python3
""" Decision Tree """
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def __str__(self):
        node_repr = f"node [feature={self.feature}, threshold={self.threshold}]"
        parts = [node_repr]
        if self.left_child:
            left_str = self.left_child_add_prefix(self.left_child.__str__())
            parts.append(left_str)
        if self.right_child:
            right_str = self.right_child_add_prefix(self.right_child.__str__())
            parts.append(right_str)
        return "\n".join(parts).strip()

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    `--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("     " + x) + "\n"
        return new_text

class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"leaf [value={self.value}] "

class Decision_Tree:
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        return self.root.__str__()
