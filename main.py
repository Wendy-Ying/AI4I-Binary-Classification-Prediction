from load import *
from linear_regression import *
from perceptron import *
from logistic_regression import *
from multi_layer_perceptron import *
from evaluation import *
import time

start_time = time.time()

X_train, Y_train, X_test, Y_test = load_dataset('ai4i2020.csv')

# linear_model = LinearRegression(n_iter=50000, lr=8e-4, batch_size=64)
# linear_model.train(X_train, Y_train, X_test, Y_test)
# linear_model.plot_loss()
# evaluate_model(linear_model, X_test, Y_test)
# plot_confusion_results(linear_model, X_test, Y_test)

# from sklearn.linear_model import LinearRegression
# linear_sklearn = LinearRegression()
# linear_sklearn.fit(X_train, Y_train)
# evaluate_model(linear_sklearn, X_test, Y_test)
# plot_confusion_results(linear_sklearn, X_test, Y_test)

# perceptron_model = Perceptron(n_iter=50000, lr=2e-3, batch_size=64)
# perceptron_model.train(X_train, Y_train, X_test, Y_test)
# perceptron_model.plot_loss()
# evaluate_model(perceptron_model, X_test, Y_test)
# plot_confusion_results(perceptron_model, X_test, Y_test)

# from sklearn.linear_model import Perceptron
# perceptron_sklearn = Perceptron(max_iter=100000, random_state=42)
# perceptron_sklearn.fit(X_train, Y_train)
# evaluate_model(perceptron_sklearn, X_test, Y_test)
# plot_confusion_results(perceptron_sklearn, X_test, Y_test)

logistic_model = LogisticRegression(n_iter=70000, lr=3e-3, batch_size=64)
logistic_model.train(X_train, Y_train, X_test, Y_test)
logistic_model.plot_loss()
evaluate_model(logistic_model, X_test, Y_test)
plot_confusion_results(logistic_model, X_test, Y_test)

# from sklearn.linear_model import LogisticRegression
# logistic_sklearn = LogisticRegression(max_iter=100000, random_state=42)
# logistic_sklearn.fit(X_train, Y_train)
# evaluate_model(logistic_sklearn, X_test, Y_test)
# plot_confusion_results(logistic_sklearn, X_test, Y_test)

# mlp_model = MultiLayerPerceptron(layer_sizes=[X_train.shape[1],12,35,73,21,2], n_iter=10000, lr=1e-6, batch_size=64)
# mlp_model.train(X_train, Y_train, X_test, Y_test)
# mlp_model.plot_loss()
# evaluate_model(mlp_model, X_test, Y_test)
# plot_confusion_results(mlp_model, X_test, Y_test)

# from sklearn.neural_network import MLPClassifier
# mlp_sklearn = MLPClassifier(hidden_layer_sizes=(12,47,101,11), activation='relu', solver='adam', max_iter=10000, random_state=42)
# mlp_sklearn.fit(X_train, Y_train)
# evaluate_model(mlp_sklearn, X_test, Y_test)
# plot_confusion_results(mlp_sklearn, X_test, Y_test)

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")