import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, n_iter=200, lr=1e-3, batch_size=128):
        self.layer_sizes = layer_sizes  # Number of neurons in each layer
        self.n_iter = n_iter  # Number of iterations (epochs)
        self.lr = lr  # Learning rate
        self.batch_size = batch_size  # Batch size for training
        self.num_layers = len(layer_sizes)  # Total number of layers
        self.weights = []  # List to store weights of each layer
        self.biases = []  # List to store biases of each layer

        # Initialize weights and biases with random values
        for i in range(self.num_layers - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def activation(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        # Derivative of sigmoid function
        return x * (1 - x)

    def forward(self, inputs):
        # Forward pass through the network
        self.activations = [inputs]  # Store activations for each layer
        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            activation = self.activation(z)
            self.activations.append(activation)
        return self.activations[-1]

    def backward(self, inputs, targets, learning_rate):
        # Ensure targets have the correct shape
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)  # Ensure targets match the output shape
        
        # Calculate output error and delta
        output_errors = targets - self.activations[-1]
        output_delta = output_errors * self.activation_derivative(self.activations[-1])
        
        # Update weights and biases for the output layer
        self.weights[-1] += np.dot(self.activations[-2].T, output_delta) * learning_rate
        self.biases[-1] += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        # Backpropagate through hidden layers
        delta = output_delta
        for i in range(self.num_layers - 2, 0, -1):
            hidden_errors = np.dot(delta, self.weights[i].T)
            hidden_delta = hidden_errors * self.activation_derivative(self.activations[i])
            
            # Update weights and biases for the hidden layer
            self.weights[i - 1] += np.dot(self.activations[i - 1].T, hidden_delta) * learning_rate
            self.biases[i - 1] += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
            delta = hidden_delta

    def mean_squared_error(self, predictions, targets):
        # Mean squared error loss
        return np.mean((predictions - targets) ** 2)

    def train(self, inputs, targets, test_inputs, test_targets):
        # Train the network using mini-batch gradient descent
        self.train_loss_history = []
        self.test_loss_history = []

        for epoch in range(self.n_iter):
            # Shuffle the training data
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            inputs_train = inputs[indices]
            targets_train = targets[indices]

            # Process data in batches
            for start in range(0, len(inputs_train), self.batch_size):
                end = min(start + self.batch_size, len(inputs_train))
                batch_inputs = inputs_train[start:end]
                batch_targets = targets_train[start:end]
                # Perform forward and backward passes
                self.forward(batch_inputs)
                self.backward(batch_inputs, batch_targets, self.lr)
            
            # Record training and testing loss
            train_predictions = self.forward(inputs)
            test_predictions = self.forward(test_inputs)
            train_loss = self.mean_squared_error(train_predictions, targets)
            test_loss = self.mean_squared_error(test_predictions, test_targets)
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)

            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.n_iter}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    def predict(self, inputs):
        # Predict class labels for inputs
        y_pred = self.forward(inputs)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        return y_pred

    def plot_loss(self):
        # Plot training and testing loss
        plt.figure()
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.plot(self.test_loss_history, label='Test Loss')
        plt.legend()
        plt.title('Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig('mlp_loss.png')
        plt.show()
