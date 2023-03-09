import numpy as np

from .Dataset import Dataset

class VarianceThreshold:

    def __init__(self, threshold: float = 0.0):

        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes    
        self.variance = None

    def fit(self, dataset: Dataset):

        # Compute the variance of each feature
        features_array = np.array(dataset.X).astype(np.float)

        self.variance = np.var(features_array, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:

        # Remove the low-variance features from the dataset
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]

        new_dataset = Dataset()
        new_dataset.X = X
        new_dataset.y= dataset.y
        new_dataset.label = dataset.label
        new_dataset.features = features

        return new_dataset

