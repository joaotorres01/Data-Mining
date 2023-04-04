from typing import Union

from collections import Counter

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, label=None, left=None, right=None):
        self.feature = feature  # index of feature to split on
        self.threshold = threshold  # threshold value to split on for numerical feature
        self.label = label  # majority label of samples in the node
        self.left = left  # left child node
        self.right = right  # right child node


class DecisionTree:
    def __init__(self, max_depth=float('inf'), min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf #the minimum number of samples required to be at a leaf node.
        self.max_leaf_nodes = max_leaf_nodes #the maximum number of leaf nodes that the tree can have. If this parameter is not None, the tree will stop growing when the number of leaf nodes reaches this value.
        self.criterion = criterion # the criterion used to select the best split. Can be 'gini', 'entropy', or 'gain_ratio'.
        self.root = None

    def __repr__(self):
        return self._repr(self.root)

    def _repr(self, node, depth=0):
        if node is None:
            return ""

        s = ""

        if node.feature is None:
            s += f"{node.label}"
        else:
            s += f"{'| ' * depth}X{node.feature} < {node.threshold}\n"
            s += self._repr(node.left, depth+1)
            s += f"{'| ' * depth}X{node.feature} >= {node.threshold}\n"
            s += self._repr(node.right, depth+1)

        return s

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Check stopping criteria
        # len(y) < self.min_samples_split: if the number of samples at the node is less than min_samples_split, the node is not split further and is turned into a leaf node.
        # depth == self.max_depth: if the current depth of the node is equal to max_depth, the node is not split further and is turned into a leaf node.
        # len(set(y)) == 1: if all the samples at the node belong to the same class, the node is turned into a leaf node.
        #(self.max_leaf_nodes is not None and len(y) <= self.max_leaf_nodes): if max_leaf_nodes is not None and the number of leaf nodes in the tree is greater than or equal to max_leaf_nodes, the node is turned into a leaf node.
        if len(y) < self.min_samples_split or depth == self.max_depth or len(set(y)) == 1 or (self.max_leaf_nodes is not None and len(y) <= self.max_leaf_nodes):
            return Node(label=self._get_majority_label(y))

        best_feature, best_threshold = self._get_best_split(X, y)


        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        # Check if the split produces enough samples
        #len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf: if the number of samples in either of the child nodes is less than min_samples_leaf, the node is not split further and is turned into a leaf node.
        if len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf:
            return Node(label=self._get_majority_label(y))

        # Recursively build left and right subtrees
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        node = Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

        # Post-pruning step
        if len(y) >= self.min_samples_split:
            error_leaf = self._error_leaf(y)
            error_decision = self._error_decision(X, y, node)
    
            # Pessimistic Error Pruning
            N = len(y)
            M = len(set(y))
            alpha = 0.5 # Pessimistic error rate parameter
            error_pessimistic = (error_leaf + alpha/N) + (alpha*M/N)
            if error_pessimistic <= error_decision:
                return Node(label=self._get_majority_label(y))
    
            # Reduced Error Pruning
            else:
                error_before_prune = self._error_decision(X, y, node)
                left_leaf = Node(label=self._get_majority_label(y[left_indices]))
                right_leaf = Node(label=self._get_majority_label(y[right_indices]))
                node.left = left_leaf
                node.right = right_leaf
                error_after_prune = self._error_decision(X, y, node)
    
                if error_after_prune <= error_before_prune:
                    return node
                else:
                    node.left = left
                    node.right = right
    
        return node
    

    def _get_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Calculate the impurity before the split
        if self.criterion == 'entropy':
            parent_impurity = self._entropy(y)
        elif self.criterion == 'gini':
            parent_impurity = self._gini_index(y)
        elif self.criterion == 'gain_ratio':
            parent_impurity = self._gain_ratio(y)
        else:
            raise ValueError(f"Unknown criterion {self.criterion}")

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                # Split the data on the threshold
                left_indices = X[:, feature_idx] < threshold
                right_indices = X[:, feature_idx] >= threshold

                # Skip if the split produces no information gain
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                # Calculate the impurity after the split
                if self.criterion == 'entropy':
                    left_impurity = self._entropy(y[left_indices])
                    right_impurity = self._entropy(y[right_indices])
                elif self.criterion == 'gini':
                    left_impurity = self._gini_index(y[left_indices])
                    right_impurity = self._gini_index(y[right_indices])
                elif self.criterion == 'gain_ratio':
                    left_impurity = self._gain_ratio(y[left_indices])
                    right_impurity = self._gain_ratio(y[right_indices])

                # Calculate the information gain
                n = len(y)
                gain = parent_impurity - (np.sum(left_indices) / n * left_impurity) - (np.sum(right_indices) / n * right_impurity)

                # Update the best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, y, left_y, right_y):
        p = len(left_y) / len(y)
        return self._entropy(y) - p * self._entropy(left_y) - (1 - p) * self._entropy(right_y)
    

    def _gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)
        
    def _information_gain(self, y, left_y, right_y):
        p = len(left_y) / len(y)
        return self._gini_index(y) - p * self._gini_index(left_y) - (1 - p) * self._gini_index(right_y)
    

    def _gain_ratio(self, y):
        class_counts = np.bincount(y)
        class_probabilities = class_counts / np.sum(class_counts)

        # Calculate the entropy of the original node
        entropy = -np.sum([p * np.log2(p) for p in class_probabilities if p > 0])

        # Calculate the information gain of each possible split
        information_gains = []
        split_information = []
        for value in np.unique(y):
            subset = y[y == value]
            proportion = len(subset) / len(y)

            # Calculate the entropy of the subset
            subset_entropy = -np.sum([(len(subset[subset == v]) / len(subset)) * np.log2(len(subset[subset == v]) / len(subset)) for v in np.unique(subset)])

            # Calculate the information gain
            information_gains.append(proportion * subset_entropy)
            split_information.append(-proportion * np.log2(proportion))

        # Calculate the split information
        split_information = np.sum(split_information)

        # Calculate the gain ratio
        gain_ratio = (entropy - np.sum(information_gains)) / split_information if split_information != 0 else 0

        return gain_ratio
    
    def _information_gain_ratio(self, y, left_y, right_y):
        p = len(left_y) / len(y)
        return self._information_gain(y, left_y, right_y) / (-p*np.log2(p) - (1-p)*np.log2(1-p))

    #majority voting
    def _get_majority_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        predictions = []
        for x in X:
            node = self.root
            while node.feature is not None:
                if x[node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.label)
        return np.array(predictions)
    
    def _error_leaf(self, y):
        """
        Calculate the error of a leaf node by counting the number of misclassifications.
        """
        num_samples = len(y)
        if num_samples == 0:
            return 0
        num_misclassified = sum(y != y[0])  # Count the number of misclassifications
        return num_misclassified / num_samples  # Calculate the error as a fraction


    def _error_decision(self, X, y, node):
        """
        Calculate the error of a decision node by measuring the impurity of its two child nodes.
        """
        # Split the data using the node's decision boundary
        mask = X[:, node.feature] <= node.threshold
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        # Calculate the weighted sum of the impurities of the child nodes
        num_left, num_right = len(y_left), len(y_right)
        impurity_left = self._gini_index(y_left)
        impurity_right = self._gini_index(y_right)
        weighted_impurity = (num_left / len(y)) * impurity_left + (num_right / len(y)) * impurity_right

        # Calculate the error as the reduction in impurity
        impurity_node = self._gini_index(y)
        error_decision = impurity_node - weighted_impurity
        return error_decision
    
        

    