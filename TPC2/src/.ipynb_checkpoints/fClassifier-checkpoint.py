from typing import Tuple, Union

import numpy as np
from scipy import stats

from .Dataset import Dataset


def f_classif(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        
        classes = np.unique(dataset.y)
        groups = [dataset.X[dataset.y == c] for c in classes]
        F, p = stats.f_oneway(*groups)
        return F, p
