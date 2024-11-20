from load import *
from linear_regression import *
from perceptron import *
from logistic_regression import *
from multi_layer_perceptron import *
from evaluation import *

X_train, Y_train, X_test, Y_test = load_dataset('ai4i2020.csv')

# linear_model = LinearRegression(n_iter=30000, lr=2e-4, batch_size=64)
# linear_model.train(X_train, Y_train, X_test, Y_test)
# linear_model.plot_loss()
# evaluate_model(linear_model, X_test, Y_test)

# perceptron_model = Perceptron(n_iter=100000, lr=7e-6, batch_size=64)
# perceptron_model.train(X_train, Y_train, X_test, Y_test)
# perceptron_model.plot_loss()
# evaluate_model(perceptron_model, X_test, Y_test)

# logistic_model = LogisticRegression(n_iter=100000, lr=1e-6, batch_size=64)
# logistic_model.train(X_train, Y_train, X_test, Y_test)
# logistic_model.plot_loss()
# evaluate_model(logistic_model, X_test, Y_test)

mlp_model = MultiLayerPerceptron(layer_sizes=[X_train.shape[1], 6, 8, 1], n_iter=1000, lr=5e-6, batch_size=64)
mlp_model.train(X_train, Y_train, X_test, Y_test)
mlp_model.plot_loss()
evaluate_model(mlp_model, X_test, Y_test)