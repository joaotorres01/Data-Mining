import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        y_pred = self.sigmoid(Z2)
        return y_pred

    def costFunction(self, X, y):
        m = y.shape[0]
        y_pred = self.predict(X)
        cost = -1/m * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
        return cost

    def buildModel(self, X, y, learning_rate=0.1, num_iterations=1000):
        m = y.shape[0]
        for i in range(num_iterations):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            y_pred = self.sigmoid(Z2)

            dZ2 = y_pred - y
            dW2 = (1/m) * np.dot(A1.T, dZ2)
            db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
            dZ1 = np.dot(dZ2, self.W2.T) * (A1 * (1 - A1))
            dW1 = (1/m) * np.dot(X.T, dZ1)
            db1 = (1/m) * np.sum(dZ1, axis=0)

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2