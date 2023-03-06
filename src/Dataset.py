from typing import Tuple, Sequence
from collections import Counter

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
    
    def read_csv(self, filename, label_col):
        df = pd.read_csv(filename)
        self.features = list(df.columns.values[:-1])
        self.label = label_col
        self.y = df[label_col].values
        df = df.drop(columns= label_col, axis=1)
        self.X = np.array(df, dtype=np.float64)
        return self

    #Using the method above
    def read_tsv(self, path, label):
        self.read_csv(path, label, '\t')

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
        means = {}
        for i, feature in enumerate(self.features):
            feature_mean = np.nanmean(self.X[:, i])
            means[feature] = feature_mean
            print(f"Média de '{feature}': {feature_mean}")
        return means

    def get_variance(self):
        variances = {}
        for i, feature in enumerate(self.features):
            feature_var = np.nanvar(self.X[:, i])
            variances[feature] = feature_var
            print(f"Variance of '{feature}': {feature_var}")
        return variances

    def get_median(self):
        medians = {}
        for i, feature in enumerate(self.features):
            feature_median = np.nanmedian(self.X[:, i])
            medians[feature] = feature_median
            print(f"Median of '{feature}': {feature_median}")
        return medians

    def get_min(self):
        mins = {}
        for i, feature in enumerate(self.features):
            feature_min = np.nanmin(self.X[:, i])
            mins[feature] = feature_min
            print(f"Minimum of '{feature}': {feature_min}")
        return mins

    def get_max(self):
        maxs = {}
        for i, feature in enumerate(self.features):
            feature_max = np.nanmax(self.X[:, i])
            maxs[feature] = feature_max
            print(f"Maximum of '{feature}': {feature_max}")
        return maxs

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


    def fill_missing_values(self):
        """
        Replaces null values with mean for numeric features and most frequent value for categorical features
        """
        for i, feature in enumerate(self.features):
            if np.issubdtype(self.X[:, i].dtype, np.number):
                feature_mean = np.nanmean(self.X[:, i])
                self.X[:, i] = np.where(np.isnan(self.X[:, i]), feature_mean, self.X[:, i])
            else:
                feature_values = self.X[:, i][~pd.isnull(self.X[:, i])]
                if len(feature_values) > 0:
                    most_frequent_value = Counter(feature_values).most_common(1)[0][0]
                    self.X[:, i] = np.where(pd.isnull(self.X[:, i]), most_frequent_value, self.X[:, i])