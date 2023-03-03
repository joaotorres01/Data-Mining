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

    #Métodos de Get e Set
    def get_X(self):
        return self.X

    def set_X(self,new):
        self.X = new

    def get_y(self):
        return self.y

    #Utilizando o método de Read CSV definido
    def read_tsv(self, path, label = None):
        self.read_csv(path, label, 't')


    def write_csv(self,filename):
            data = pd.DataFrame(data=np.column_stack((self.X, self.y)), columns=self.features+[self.label])
            data.to_csv(filename, index=False)

    def write_tsv(self,filename):
            data = pd.DataFrame(data=np.column_stack((self.X, self.y)), columns=self.features+[self.label])
            data.to_tsv(filename, index=False)

    def count_nulls(self):
            null_counts = {}
            for i, feature in enumerate(self.features):
                null_counts[feature] = np.sum(pd.isnull(self.X[:, i]))

            null_counts[self.label] = np.sum(pd.isnull(self.y))
            return null_counts

    def summary(self):
            print("Number of instances:", len(self))
            print("Number of features:", len(self.features))
            print("Feature names:", self.features)
            print("Label name:", self.label)
            print("X: ", self.X, "\n")
            print("y: ", self.y, "\n")