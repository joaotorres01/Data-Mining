import numpy as np

class Prism:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def fit(self, X, y):
        self.rules = []
        for i in range(X.shape[1]):
            vals = np.unique(X[:,i])
            for val in vals:
                mask = X[:,i] == val
                count = np.bincount(y[mask])
                if len(count) == 0:
                    continue
                max_class = np.argmax(count)
                max_count = count[max_class]
                prob = max_count/len(y[mask])
                if prob >= self.threshold:
                    rule = (i, val, max_class, prob)
                    self.rules.append(rule)
    
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            max_prob = -1
            max_class = None
            for rule in self.rules:
                feature_idx, feature_val, class_val, prob = rule
                if X[i, feature_idx] == feature_val and prob > max_prob:
                    max_prob = prob
                    max_class = class_val
            if max_class is not None:
                y_pred.append(max_class)
            else:
                y_pred.append(0)
        return np.array(y_pred)
    
    def __repr__(self):
        return str(self.rules)