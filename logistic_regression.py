import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, n_iter=200, lr=1e-3, alpha=0.05, batch_size=64):
        self.n_iter = n_iter # number of iterations
        self.lr = lr # learning rate
        self.alpha = alpha # regularization parameter
        self.batch_size = batch_size # batch size
        self.W = None # weights
        self.loss = [] # training loss
        self.test_loss = [] # test loss

    @staticmethod
    def _sigmoid(z):
        # sigmoid function
        return 1 / (1 + np.exp(-z))

    def predict_probability(self, X):
        return self._sigmoid(X @ self.W)

    @staticmethod
    def _loss(y, y_pred, epsilon=1e-5):
        # Weighted cross entropy loss
        weights = np.where(y == 1, 0.7, 0.3)
        loss = -weights * (y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return np.mean(loss)

    def _gradient(self, X, y, y_pred):
        # Weighted gradient for cross entropy loss
        weights = np.where(y == 1, 0.7, 0.3)
        reg_term = self.alpha * self.W  # Regularization term
        weighted_diff = weights * (y_pred - y)
        return (weighted_diff @ X) / y.size + reg_term

    @staticmethod
    def preprocess_data(X):
        m, n = X.shape
        X_new = np.hstack((np.ones((m, 1)), X))  # Add bias term
        return X_new

    def _train_single_epoch(self, X, y):
        indices = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], self.batch_size):
            batch_indices = indices[i:min(i + self.batch_size, X.shape[0])]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            y_pred = self.predict_probability(X_batch)
            grad = self._gradient(X_batch, y_batch, y_pred)
            self.W -= self.lr * grad

    def train(self, X_train, y_train, X_test, y_test):
        # add bias term
        X_train = self.preprocess_data(X_train)
        X_test = self.preprocess_data(X_test)
        self.W = np.random.rand(X_train.shape[1])

        for _ in range(self.n_iter):
            self._train_single_epoch(X_train, y_train)
            # Record training loss
            y_train_pred = self.predict_probability(X_train)
            self.loss.append(self._loss(y_train, y_train_pred))
            # Record test loss if test data is provided
            y_test_pred = self.predict_probability(X_test)
            self.test_loss.append(self._loss(y_test, y_test_pred))

    def predict(self, X):
        X = self.preprocess_data(X)
        y_pred = self.predict_probability(X)
        return np.where(y_pred >= 0.5, 1, 0)

    def plot_loss(self):
        plt.plot(self.loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss')
        plt.legend()
        plt.title('Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig('logistic_loss.png')
        plt.show()
