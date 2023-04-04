from typing import Tuple, Union

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

import numpy as np

from .Dataset import Dataset


def f_regress(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        
        F, p = f_regression(dataset.X, dataset.y)
        return F, p