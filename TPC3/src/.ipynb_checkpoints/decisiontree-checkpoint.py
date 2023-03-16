from .Dataset import Dataset

import numpy as np


class DecisionTreeNode:
    """
    Base class for decision tree nodes.
    """
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        self.split_attr = None
        self.child_nodes = {}
    
    def split(self, split_attr):
        """
        Splits the dataset into subsets based on the given attribute and creates child nodes for each subset.
        """
        raise NotImplementedError()
    
    def predict(self, row):
        """
        Returns the predicted class for the given row of data.
        """
        raise NotImplementedError()


class DecisionTreeRootNode(DecisionTreeNode):
    """
    Class for the root node of a decision tree.
    """
    def __init__(self, dataset, target):
        super().__init__(dataset, target)
        self.split_attr = self.select_best_attribute()
    
    def select_best_attribute(self):
        """
        Returns the best attribute to split on based on a criterion such as information gain.
        """
        raise NotImplementedError()


class DecisionTreeNonLeafNode(DecisionTreeNode):
    """
    Class for a non-leaf node of a decision tree.
    """
    def __init__(self, dataset, target, split_attr):
        super().__init__(dataset, target)
        self.split_attr = split_attr
    
    def split(self, split_attr):
        """
        Splits the dataset into subsets based on the given attribute and creates child nodes for each subset.
        """
        self.child_nodes = {}
        for val in self.dataset[split_attr].unique():
            subset = self.dataset[self.dataset[split_attr] == val].drop(split_attr, axis=1)
            child_node = DecisionTreeLeafNode(subset, self.target, self.predict(subset))
            self.child_nodes[val] = child_node
    
    def predict(self, row):
        """
        Returns the predicted class for the given row of data.
        """
        val = row[self.split_attr]
        child_node = self.child_nodes[val]
        return child_node.predict(row)


class DecisionTreeLeafNode(DecisionTreeNode):
    """
    Class for a leaf node of a decision tree.
    """
    def __init__(self, dataset, target, predicted_class):
        super().__init__(dataset, target)
        self.predicted_class = predicted_class
    
    def predict(self, row):
        """
        Returns the predicted class for the given row of data.
        """
        return self.predicted_class

class DecisionTree:
    """
    Class for creating and using decision trees.
    """
    def __init__(self, criterion="entropy"):
        self.root_node = None
        self.criterion = criterion
    
    def fit(self, X, y):
        """
        Fits the decision tree to the given training data.
        """
        self.root_node = self.build_tree(X, y)
    
    def predict(self, X):
        """
        Predicts the class labels for the given data using the decision tree.
        """
        y_pred = []
        for _, row in X.iterrows():
            y_pred.append(self.predict_row(row))
        return y_pred
    
    def build_tree(self, X, y):
        """
        Recursively builds a decision tree on the given data.
        """
        if len(set(y)) == 1:
            # If all examples have the same label, create a leaf node with that label
            return DecisionTreeLeafNode(X, y, y[0])
        else:
            # Otherwise, create a non-leaf node and split the data on the best attribute
            node = DecisionTreeNonLeafNode(X, y, None)
            node.split_attr = self.select_best_attribute(X, y)
            node.split(node.split_attr)
            return node
    
    def predict_row(self, row):
        """
        Predicts the class label for a single row of data.
        """
        node = self.root_node
        while not isinstance(node, DecisionTreeLeafNode):
            val = row[node.split_attr]
            node = node.child_nodes[val]
        return node.predicted_class
    
    def select_best_attribute(self, X, y):
        """
        Selects the best attribute to split on based on the given criterion.
        """
        if self.criterion == "entropy":
            # Calculate entropy for each attribute and choose the one with the highest information gain
            return self.select_best_attribute_entropy(X, y)
        elif self.criterion == "gini":
            # Calculate Gini impurity for each attribute and choose the one with the highest Gini gain
            return self.select_best_attribute_gini(X, y)
        else:
            raise ValueError(f"Unknown criterion '{self.criterion}'")
    
    def select_best_attribute_entropy(self, X, y):
        """
        Selects the best attribute to split on based on information gain.
        """
        entropy = self.calculate_entropy(y)
        max_gain = -float("inf")
        best_attr = None
        for attr in X.columns:
            gain = entropy - self.calculate_conditional_entropy(X[attr], y)
            if gain > max_gain:
                max_gain = gain
                best_attr = attr
        return best_attr
    
    def select_best_attribute_gini(self, X, y):
        """
        Selects the best attribute to split on based on Gini impurity.
        """
        gini = self.calculate_gini(y)
        max_gain = -float("inf")
        best_attr = None
        for attr in X.columns:
            gain = gini - self.calculate_conditional_gini(X[attr], y)
            if gain > max_gain:
                max_gain = gain
                best_attr = attr
        return best_attr
    
    def calculate_entropy(self, y):
        """
        Calculates the entropy of the given target variable.
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))
    
    def calculate_gini(self, y):
        """
        Calculates the Gini impurity of the given target variable.
        """
        values, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def calculate_conditional_entropy(self, X_attr, y):
        """
        Calculates the conditional entropy of the given attribute.
        """
        values, counts = np.unique(X_attr, return_counts=True)
        probs = counts / len(X_attr)
        entropies = [self.calculate_entropy(y[X_attr == v]) for v in values]
        return np.sum(probs * entropies)
    
    def calculate_conditional_gini(self, X_attr, y):
        """
        Calculates the conditional Gini impurity of the given attribute.
        """
        values, counts = np.unique(X_attr, return_counts=True)
        probs = counts / len(X_attr)
        ginis = [self.calculate_gini(y[X_attr == v]) for v in values]
        return np.sum(probs * ginis)
