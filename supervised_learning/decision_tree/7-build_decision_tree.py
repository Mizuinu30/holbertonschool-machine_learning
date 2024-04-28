#!/usr/bin/env python3
"""depth of a decision tree"""
import numpy as np


class Node:
    """ represent a node in a decision tree"""
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
        """calculate the maximum depth below the current node"""
        if self.is_leaf:
            return self.depth
        else:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Calculate the number of nodes below this node"""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        else:
            return (1 + self.left_child.count_nodes_below() +
                    self.right_child.count_nodes_below())

    def left_child_add_prefix(self, text):
        """print the left child with the correct prefix
        split at line breaks, add spaces, +, --, |,
        and then join the lines back together"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """print the right child with the correct prefix
        split at line breaks, add spaces, +, --, but no |
        and then join the lines back together"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0]
        for x in lines[1:]:
            new_text += "\n       " + x
        return new_text

    def __str__(self):
        """print root or node with feature and threshold
        then print left and right children"""
        if self.is_root:
            node_text = (
                f"root [feature={self.feature},"
                f" threshold={self.threshold}]"
            )
        else:
            node_text = (
                f"-> node [feature={self.feature},"
                f" threshold={self.threshold}]"
            )

        left_child_str = self.left_child_add_prefix(str(self.left_child))
        right_child_str = self.right_child_add_prefix(str(self.right_child))
        return f"{node_text}\n{left_child_str}{right_child_str}"

    def get_leaves_below(self):
        """Get all the leaves below this node."""
        return (self.left_child.get_leaves_below()
                + self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Update the bounds of the leaves below the current node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = max(
                        child.lower.get(self.feature, -np.inf), self.threshold)
                else:  # right child
                    child.upper[self.feature] = min(
                        child.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """Update the indicator function of the leaves
        below the current node."""

        def is_large_enough(x):
            """returns a 1Darray of size `n_individuals`
            so that the `i`-th element of the later is `True`
            if the `i`-th individual has all its
            features > the lower bounds"""
            lower_bounds = np.array([self.lower.get(i, -np.inf)
                                     for i in range(x.shape[1])])
            return np.all(x > lower_bounds, axis=1)

        def is_small_enough(x):
            """returns a 1Dnumoy array of size `n_individuals`
            so that the `i`-th element of the later is `True`
            if the `i`-th individual has all its
            features <= the lower bounds"""
            upper_bounds = np.array([self.upper.get(i, np.inf)
                                     for i in range(x.shape[1])])
            return np.all(x <= upper_bounds, axis=1)

        """returns a 1Dboolean array where ith element is True if
        both is_large_enough and is_small_enough are True for the
        ith individual"""
        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """predict the value of a data point"""
        if x[self.feature] > self.threshold:
            return self.right_child.pred(x)
        else:
            return self.left_child.pred(x)


class Leaf(Node):
    """representing a leaf in a decision tree"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """calculate the maximum depth below the current node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Calculate the number of nodes below this node."""
        return 1

    def __str__(self):
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """Get all the leaves below this node."""
        return [self]

    def update_bounds(self, bounds):
        """update the bounds of the leaf"""
        pass

    def pred(self, x):
        """predict the value of a data point"""
        return self.value


class Decision_Tree():
    """representing a decision tree"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
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

    def update_bounds(self):
        """update the bounds of the leaves in the tree"""
        self.root.update_bounds_below()

    def update_predict(self):
        """update the predict method of the decision tree"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def pred(self, x):
        """predict the value of a data point"""
        return self.root.pred(x)

    def fit(self,explanatory, target,verbose=0):
        """ Fit the decision tree"""
        if self.split_criterion == "random" :
                self.split_criterion = self.random_split_criterion
        else :
                self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target      = target
        self.root.sub_population = np.ones_like(self.target,dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose==1 :
                print(f"""  Training finished.
- Depth                     : { self.depth()       }
- Number of nodes           : { self.count_nodes() }
- Number of leaves          : { self.count_nodes(only_leaves=True) }
- Accuracy on training data : { self.accuracy(self.explanatory,self.target)    }""")

    def np_extrema(self,arr):
        """ Return the minimum and maximum of an array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self,node):
        """ Randomly select a feature and a threshold"""
        diff=0
        while diff==0 :
            feature=self.rng.integers(0,self.explanatory.shape[1])
            feature_min,feature_max=self.np_extrema(self.explanatory[:,feature][node.sub_population])
            diff=feature_max-feature_min
        x=self.rng.uniform()
        threshold= (1-x)*feature_min + x*feature_max
        return feature,threshold

    def fit_node(self, node):
        """ Fit a node in the decision tree"""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = self.explanatory[node.sub_population, node.feature] <= node.threshold
        right_population = self.explanatory[node.sub_population, node.feature] > node.threshold

        # Is left node a leaf?
        is_left_leaf = np.sum(left_population) < self.min_pop or node.depth >= self.max_depth or np.all(self.target[node.sub_population][left_population] == self.target[node.sub_population][left_population][0])

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        is_right_leaf = np.sum(right_population) < self.min_pop or node.depth >= self.max_depth or np.all(self.target[node.sub_population][right_population] == self.target[node.sub_population][right_population][0])

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """ Return a leaf node"""
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child= Leaf( value )
        leaf_child.depth=node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ Return a node child"""
        n= Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory , test_target):
        """ Return the accuracy of the decision tree on the test data"""
        return np.sum(np.equal(self.predict(test_explanatory), test_target))/test_target.size
