Decision Tree & Random Forest

Task 0: Depth of a Decision Tree

    Objective: Implement a method to calculate the maximum depth of a decision tree.
    Details: Update the Node class to include a method max_depth_below(self), which calculates the maximum depth below the current node.

Task 1: Number of Nodes/Leaves in a Decision Tree

    Objective: Count the number of nodes and leaves in a decision tree.
    Details: Implement the method count_nodes_below(self, only_leaves=False) in the Node class to count all nodes or just the leaves, depending on the parameter.

Task 2: Let's Print Our Tree

    Objective: Implement tree printing functionality.
    Details: Add a __str__ method to the Node and Leaf classes to allow for a readable string representation of the tree structure.

Task 3: Towards the Predict Function (1) - The Get Leaves Method

    Objective: Implement a method to retrieve all leaves of the tree.
    Details: Update the Node class to include a method get_leaves_below(self) that returns all leaves below the current node.

Task 4: Towards the Predict Function (2) - The Update Bounds Method

    Objective: Implement a method to update the bounds for each node.
    Details: Complete the update_bounds_below(self) method in the Node class to compute and update the bounds for each node's features.

Task 5: Towards the Predict Function (3) - The Update Indicator Method

    Objective: Implement a method to update the indicator function for a node.
    Details: Complete the update_indicator(self) method in the Node class, which updates the indicator function based on feature bounds.

Task 6: The Predict Function

    Objective: Implement the prediction functionality for the decision tree.
    Details: Develop the update_predict(self) method in the Decision_Tree class to enable prediction based on the trained tree.

Task 7: Training Decision Trees

    Objective: Implement the functionality to train the decision tree with a dataset.
    Details: Develop the fit(self, explanatory, target) method in the Decision_Tree class, allowing the tree to learn from data.

Task 8: Using Gini Impurity Function as a Splitting Criterion

    Objective: Implement the Gini impurity function to enhance the decision tree's splitting process.
    Details: Update the Decision_Tree class to include methods that calculate the best split based on Gini impurity for a given feature.

Task 9: Random Forests

    Objective: Implement a Random Forest classifier.
    Details: Develop the Random_Forest class with methods to train multiple decision trees and predict output by aggregating their predictions.

Task 10: IRF 1 - Isolation Random Trees

    Objective: Apply the concept of random forests to outlier detection using isolation trees.
    Details: Implement the Isolation_Random_Tree class for detecting outliers based on the structure of random trees.

Task 11: IRF 2 - Isolation Random Forests

    Objective: Create an isolation forest for identifying potential outliers in a dataset.
    Details: Develop the Isolation_Random_Forest class, which uses multiple isolation trees to predict the 'outlierness' of each data point.