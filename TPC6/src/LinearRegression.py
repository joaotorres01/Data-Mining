import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, dataset, normalize=False, regularization=False, lamda=1):
        self.X, self.y = dataset.get_X(), dataset.get_y()
        self.X = np.hstack((np.ones([self.X.shape[0], 1]), self.X))
        self.theta = np.zeros(self.X.shape[1])
        self.regularization = regularization
        self.lamda = lamda
        if normalize: 
            self.normalize()
        else: 
            self.normalized = False

    def buildModel(self):
        from numpy.linalg import inv
        if self.regularization:
            self.analyticalWithReg()    
        else:
            self.theta = inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)

    def analyticalWithReg(self):
        from numpy.linalg import inv
        matl = np.zeros([self.X.shape[1], self.X.shape[1]])
        for i in range(1, self.X.shape[1]): 
            matl[i, i] = self.lamda
        mattemp = inv(self.X.T.dot(self.X) + matl)
        self.theta = mattemp.dot(self.X.T).dot(self.y)

    def predict(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])

        if self.normalized:
            x[1:] = (x[1:] - self.mu) / self.sigma 
        return np.dot(self.theta, x)

    def costFunction(self):
        m = self.X.shape[0]
        predictions = np.dot(self.X, self.theta)
        sqe = (predictions - self.y) ** 2
        res = np.sum(sqe) / (2 * m)
        return res

    def gradientDescent(self, iterations=1000, alpha=0.001):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)
        if self.regularization:
            lamdas = np.zeros([self.X.shape[1]])
            for i in range(1, self.X.shape[1]): 
                lamdas[i] = self.lamda
        for its in range(iterations):
            J = self.costFunction()
            if its % 100 == 0: 
                print(J)
            delta = self.X.T.dot(self.X.dot(self.theta) - self.y)                      
            if self.regularization:
                self.theta -= (alpha / m * (lamdas + delta))
            else: 
                self.theta -= (alpha / m * delta )
            
    def printCoefs(self):
        print(self.theta)

    def plotData_2vars(self, xlab, ylab):
        plt.plot(self.X[:, 1], self.y, 'rx', markersize=7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.show()
        
    def plotDataAndModel(self, xlab, ylab):
        fig, ax = plt.subplots()

        # Plot the training data
        ax.plot(self.X[:, 1], self.y, 'rx', markersize=7, label='Training data')
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)

        # Sort the indices based on the first feature
        sorted_indices = np.argsort(self.X[:, 1])

        # Plot the linear regression line
        ax.plot(self.X[sorted_indices, 1], np.dot(self.X[sorted_indices], self.theta[1:]), '-',
                label='Linear regression')

        ax.legend()
        plt.show()


    def normalize(self):
        self.mu = np.mean(self.X[:, 1:], axis=0)
        self.X[:, 1:] = self.X[:, 1:] - self.mu
        self.sigma = np.std(self.X[:, 1:], axis=0)
        self.X[:, 1:] = self.X[:, 1:] / self.sigma
        self.normalized = True