import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, n_iter=200, lr=1e-3, batch_size=128):
        self.n_iter = n_iter
        self.lr = lr
        self.batch_size = batch_size
        self.W = None
        self.train_loss = []
        self.test_loss = []
    
    def preprocess_data_X(self, X):
        # add bias term to X
        m, n = X.shape
        X_ = np.empty([m, n + 1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def predict(self, X):
        # predict y
        X = self.preprocess_data_X(X)
        return X @ self.W
    
    def plot_loss(self):
        # plot loss
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss', linestyle='--')
        # set title and labels
        plt.title("Loss over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig('linear_loss.png')
        plt.show()

    def calculate_loss(self, y_true, y_pred):
        # MSE loss
        weights = np.where(y_true == 0, 0.4, 0.6)
        loss = weights * (y_true - y_pred)**2
        return np.mean(loss)

    def gradient(self, X, y, y_pred):
        # gradient
        X = self.preprocess_data_X(X)
        weights = np.where(y == 0, 0.4, 0.6) 
        grad = (X.T @ ((weights * (y_pred - y)).reshape(-1, 1))).flatten() / y.size
        return grad

    def train(self, X_train, y_train, X_test, y_test):
        # initialize weights
        self.W = np.random.rand(X_train.shape[1] + 1)
        N = y_train.size
        for _ in range(self.n_iter):
            # shuffle the data
            indices = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)  # not out of range
                # get batch data
                batch_indices = indices[start:end]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                # predict and compute loss
                y_pred = self.predict(X_batch)
                train_loss = self.calculate_loss(y_batch, y_pred)
                self.train_loss.append(train_loss)
                y_pred_test = self.predict(X_test)
                test_loss = self.calculate_loss(y_test, y_pred_test)
                self.test_loss.append(test_loss)
                # compute gradient
                grad = self.gradient(X_batch, y_batch, y_pred)
                self.W -= self.lr * grad
                
    