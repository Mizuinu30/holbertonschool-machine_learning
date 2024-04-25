#!/usr/bin/env python3
""" Module that builds a decision tree """


import numpy as np


class Node:
    """ Node of the decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
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
            left_depth = self.left_child.max_depth_below()
            local_max = max(local_max, left_depth)

        if self.right_child:
            right_depth = self.right_child.max_depth_below()
            local_max = max(local_max, right_depth)

        return local_max

    def left_child_add_prefix(self,text):
        """ Add prefix to the left child"""
        lines=text.split("\n")
        new_text="    +--"+lines[0] + "\n"
        for x in lines[1:] :
            new_text+=("    |  "+x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """ Add prefix to the right child"""
        lines = text.split("\n")
        new_text = "    --+"+lines[0]+"\n"
        for x in lines[1:] :
            new_text+=("     "+x)+"\n"
        return (new_text)

    def count_nodes_below(self, only_leaves=False):
        """ Count the number of nodes below this node"""
        if only_leaves:
            count = 0
        else:
            count = 1

        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves=only_leaves)

        if self.right_child:
            count += self.right_child.count_nodes_below\
                (only_leaves=only_leaves)

        return count


class Leaf(Node):
    """ Leaf node of the decision tree"""
    def __init__(self, value, depth=None):
        """ Constructor"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """ Return the string representation of the node"""
        if self.is_leaf:
            return (f"-> leaf [value={self.value}] ")
        else:
            result = (f"-> node [feature={self.feature}, threshold={self.threshold}] ")
            if self.left_child:
                result += "\n    +-- " + str(self.left_child).replace("\n", "\n    |  ")
            if self.right_child:
                result += "\n    +-- " + str(self.right_child).replace("\n", "\n       ")
            return result

    def max_depth_below(self):
        """ Calculate the maximum depth below this node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Count the number of nodes below this node"""
        return 1


class Decision_Tree():
    """ Class that builds a decision tree"""
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
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
        """ String representation of the decision tree"""
        return self.root.__str__()

    def depth(self):
        """ Calculate the depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Count the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)
