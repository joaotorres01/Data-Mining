from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

import numpy as np

from .Dataset import Dataset

class f_Regression:
    
    def __init__(self, dataset: Dataset):
        self.X = dataset.X
        self.y = dataset.y
        self.p_values = self.get_p_values()
        
    def get_p_values(self):
        f_values, p_values = f_regression(self.X, self.y)
        return p_values