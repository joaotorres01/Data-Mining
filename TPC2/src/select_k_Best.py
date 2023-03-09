from typing import Callable

import numpy as np

from .Dataset import Dataset
from .fClassifier import f_classif


class Select_K_Best:

    def __init__(self, score_func: Callable = f_classif, k: int = 10):
        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'Select_K_Best':

        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:

        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]

        new_dataset = Dataset()
        new_dataset.X = dataset.X[:, idxs]
        new_dataset.y= dataset.y
        new_dataset.label = dataset.label
        new_dataset.features = features=list(features)

        return new_dataset

    def fit_transform(self, dataset: Dataset) -> Dataset:

        self.fit(dataset)
        return self.transform(dataset)