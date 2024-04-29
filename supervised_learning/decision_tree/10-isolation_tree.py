#!/usr/bin/env python3
""" Module that builds an isolation tree """


import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """ Random Isolation Tree """
    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """ String representation of the tree """
        return self.root.__str__()

    def depth(self):
        """ Calculate the depth of the tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Count the number of nodes in the tree """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """ Get all the leaves in the tree """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Update the bounds of the tree """
        self.root.update_bounds_below()

    def update_predict(self):
        """ Update the prediction of the tree """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

    def np_extrema(self, arr):
        """ Calculate the extrema of an array """
        return np.min(arr), np.max(arr)

    def randoms_split_criterion(self, node):
        """ Random split criterion """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """ Get the leaf children """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ Get the node children """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """ Recursive fit """
        node.feature, node.threshold = self.random_split_criterion(node)

        above_threshold = self.explanatory[:, node.feature] > node.threshold
        left_population = node.sub_population & above_threshold
        right_population = node.sub_population & ~above_threshold

        is_left_leaf = np.any([
            node.depth >= self.max_depth - 1,
            np.sum(left_population) <= self.min_pop
        ])

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = np.any([
            node.depth >= self.max_depth - 1,
            np.sum(right_population) <= self.min_pop
        ])

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """ Fit the isolation tree """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0],
                                           dtype=bool)
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f""" Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }""")
