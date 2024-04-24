#!/usr/bin/env python3
""" Module that builds a decision tree """


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
