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
    def __init__(self, max_depth=float('inf')):
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Check stopping criteria
        if depth >= self.max_depth or len(y) == 0:
            return Node(label=self._get_majority_label(y))
        if len(set(y)) == 1:
            return Node(label=y[0])

        # Find best split
        best_feature, best_threshold = self._get_best_split(X, y)
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        # Recursively build left and right subtrees
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _get_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float('inf')

        for feature in range(X.shape[1]):
            # Get all unique values for the feature
            feature_values = set(X[:, feature])
            for threshold in feature_values:
                # Split samples based on threshold
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                # Calculate gini impurity of split
                left_gini = self._gini_impurity(y[left_indices])
                right_gini = self._gini_impurity(y[right_indices])
                gini = (sum(left_indices) / len(y)) * left_gini + (sum(right_indices) / len(y)) * right_gini

                # Update best split if necessary
                if gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = gini

        return best_feature, best_threshold

    def _gini_impurity(self, y):
        p = [np.sum(y == c) / len(y) for c in set(y)]
        return 1 - sum([x**2 for x in p])

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