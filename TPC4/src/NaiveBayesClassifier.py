import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y, alpha=1):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.mean = np.zeros((len(self.classes), n_features), dtype=np.float64)
        self.var = np.zeros((len(self.classes), n_features), dtype=np.float64)
        self.priors = np.zeros(len(self.classes), dtype=np.float64)

        for idx, label in enumerate(self.classes):
            X_c = X[y==label]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + alpha
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])


    def predict(self, X, epsilon=1e-9):
        y_pred = np.zeros(X.shape[0], dtype=self.classes.dtype)
        for i, x in enumerate(X):
            posteriors = []
            for j, label in enumerate(self.classes):
                prior = np.log(self.priors[j])
                class_conditional = np.sum(np.log(self.pdf(j, np.array(x), epsilon)))
                posterior = prior + class_conditional
                posteriors.append(posterior)
            y_pred[i] = self.classes[np.argmax(posteriors)]
        return y_pred
    
    def pdf(self, class_idx, x, epsilon=1e-9):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x-mean)**2)/(2*var))
        denominator = np.sqrt(2*np.pi*var) + epsilon
        return numerator / denominator