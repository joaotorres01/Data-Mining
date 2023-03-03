from typing import Tuple, Sequence

import pandas as pd
import numpy as np


class Dataset:
    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])
        self.features = []
        self.label = ''

    def __len__(self):
        if self.X is not None:
            return len(self.X)
        else:
            return 0
    
    def read_csv(self, filename, label_col=-1):
        df = pd.read_csv(filename)
        self.features = list(df.columns.values[:-1])
        self.label = df.columns.values[-1]
        self.y = df.iloc[:, label_col].values
        df = df.drop(df.columns[label_col], axis=1)
        self.X = np.array(df, dtype=object)
        return self

    #Using the method above
    def read_tsv(self, path, label = None):
        self.read_csv(path, label, 't')

    #Get and Set
    def get_X(self):
        return self.X

    def set_X(self,new):
        self.X = new

    def get_y(self):
        return self.y

        
    #Write to file
    def write_csv(self,filename):
            data = pd.DataFrame(data=np.column_stack((self.X, self.y)), columns=self.features+[self.label])
            data.to_csv(filename, index=False)

    def write_tsv(self,filename):
            data = pd.DataFrame(data=np.column_stack((self.X, self.y)), columns=self.features+[self.label])
            data.to_tsv(filename, index=False)

    def get_mean(self):
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    #Basic info
    def summ(self):
            print("Number of features:", len(self.features))
            print("Number of instances:", len(self))
            print("Names of the features :", self.features)
            print("Name of the label:", self.label)
            print("X: ", self.X, "\n")
            print("y: ", self.y, "\n")

    #Describe
    def describe(self):
        if self.X is not None:
            df = pd.DataFrame(data=self.X, columns=self.features)
            return df.describe()
        else:
            return None

    #Count the null values
    def nullcount(self):
        counter = {}
        for i, feature in enumerate(self.features):
            counter[feature] = np.sum(pd.isnull(self.X[:, i]))
            
        counter[self.label] = np.sum(pd.isnull(self.y))
        return counter