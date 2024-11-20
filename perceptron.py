import numpy as np
import matplotlib.pyplot as plt

def convert_labels(y):
    # Convert labels to -1 and 1
    y = np.array(y)
    return np.where(y == 0, -1, 1)

class Perceptron:
    def __init__(self, n_iter=200, lr=0.01, batch_size=64):
        self.n_iter = n_iter  # Number of iterations
        self.lr = lr  # Learning rate
        self.batch_size = batch_size  # Batch size
        self.W = None  # Initialize weights
        self.loss = []  # Training loss history
        self.test_loss = []  # Test loss history

    def _loss_batch(self, y, y_pred):
        # Weighted hinge loss for a batch
        weights = np.where(y == 1, 0.5, 0.5)
        loss = np.maximum(0, -y * y_pred) * weights
        return loss.mean()

    def _gradient_batch(self, X, y, y_pred):
        # Gradient of weighted hinge loss for a batch
        weights = np.where(y == 1, 0.5, 0.5)
        misclassified = y_pred * y < 0
        gradient = -(X[misclassified].T @ (weights[misclassified] * y[misclassified])) / X.shape[0]
        return gradient

    def _preprocess_data(self, X):
        # Add bias term to the feature matrix
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def mbgd_update(self, X, y, X_test, y_test):
        # Mini-batch gradient descent
        n_samples = X.shape[0]
        self.W = np.random.rand(X.shape[1])

        for epoch in range(self.n_iter):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            for start in range(0, n_samples, self.batch_size):
                # Mini-batch data
                end = start + self.batch_size
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                # Compute predictions and gradients
                y_pred = X_batch @ self.W
                grad = self._gradient_batch(X_batch, y_batch, y_pred)

                # Update weights
                self.W -= self.lr * grad

            # Calculate loss for the epoch
            train_loss = self._loss_batch(y, X @ self.W)
            test_loss = self._loss_batch(y_test, X_test @ self.W)
            self.loss.append(train_loss)
            self.test_loss.append(test_loss)

    def train(self, X_train, y_train, X_test, y_test):
        # Train the perceptron
        X_train = self._preprocess_data(X_train)
        X_test = self._preprocess_data(X_test)
        y_train = convert_labels(y_train)
        y_test = convert_labels(y_test)
        self.mbgd_update(X_train, y_train, X_test, y_test)

    def predict(self, X):
        # Predict labels for input data
        X = self._preprocess_data(X)
        y = X @ self.W
        y = np.where(y >= 0, 1, 0)
        return y

    def plot_loss(self):
        # Plot training and test loss over epochs
        plt.plot(self.loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss')
        plt.title('Loss over Iterations')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig('perceptron_loss.png')
        plt.show()
