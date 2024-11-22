# AI4I Binary Classification Prediction
## linear regression
### my model
linear_model = LinearRegression(n_iter=50000, lr=8e-4, batch_size=64)

**TP: 83  TN: 206  FP: 36  FN: 21**
Accuracy: 0.8352601156069365
Precision: 0.6974789915966386
Recall: 0.7980769230769231
F1 Score: 0.7443946188340808
Total time taken: 38.09750461578369 seconds

<img src="./linear_loss-1.png" width=45%> <img src="./confusion_results-1.png" width=45%>

### sklearn
linear_sklearn = LinearRegression()

**TP: 78  TN: 217  FP: 20  FN: 24**
Accuracy: 0.8702064896755162
Precision: 0.7959183673469388
Recall: 0.7647058823529411
F1 Score: 0.7799999999999999
Total time taken: 5.636926174163818 seconds

<img src="./confusion_results-2.png" width=45%>

## perceptron
### my model
perceptron_model = Perceptron(n_iter=50000, lr=2e-3, batch_size=64)

**TP: 102  TN: 242  FP: 44  FN: 21**
Accuracy: 0.8410757946210269
Precision: 0.6986301369863014
Recall: 0.8292682926829268
F1 Score: 0.758364312267658
Total time taken: 19.484343767166138 seconds

<img src="./perceptron_loss-1.png" width=45%> <img src="./confusion_results-3.png" width=45%>

### sklearn
perceptron_sklearn = Perceptron(max_iter=100000, random_state=42)

**TP: 94  TN: 232  FP: 36  FN: 21**
Accuracy: 0.8511749347258486
Precision: 0.7230769230769231
Recall: 0.8173913043478261
F1 Score: 0.7673469387755102
Total time taken: 2.7166144847869873 seconds

<img src="./confusion_results-4.png" width=45%>

## logistic regression
### my model
logistic_model = LogisticRegression(n_iter=30000, lr=3e-3, batch_size=64)

**TP: 85  TN: 195  FP: 52  FN: 21**
Accuracy: 0.7932011331444759
Precision: 0.6204379562043796
Recall: 0.8018867924528302
F1 Score: 0.6995884773662552
Total time taken: 19.374540090560913 seconds

<img src="./logistic_loss-1.png" width=45%> <img src="./confusion_results-5.png" width=45%>

### sklearn
logistic_sklearn = LogisticRegression(max_iter=100000, random_state=42)

**TP: 61  TN: 165  FP: 26  FN: 21**
Accuracy: 0.8278388278388278
Precision: 0.7011494252873564
Recall: 0.7439024390243902
F1 Score: 0.7218934911242605
Total time taken: 2.442545175552368 seconds

<img src="./confusion_results-6.png" width=45%>

## MLP
### my model
mlp_model = MultiLayerPerceptron(layer_sizes=[X_train.shape[1],47,101,32], n_iter=10000, lr=1e-5, batch_size=32)

**TP: 87  TN: 213  FP: 20  FN: 13**
Accuracy: 0.9009009009009009
Precision: 0.8130841121495327
Recall: 0.87
F1 Score: 0.8405797101449274
Total time taken: 421.797244310379 seconds

<img src="./mlp_loss-1.png" width=45%> <img src="./confusion_results-7.png" width=45%>

### sklearn
mlp_sklearn = MLPClassifier(hidden_layer_sizes=(12,47,101,11), activation='relu', solver='adam', max_iter=10000, random_state=42)

**TP: 90  TN: 213  FP: 15  FN: 8**
Accuracy: 0.9294478527607362
Precision: 0.8571428571428571
Recall: 0.9183673469387755
F1 Score: 0.8866995073891625
Total time taken: 8.705824136734009 seconds

<img src="./confusion_results-8.png" width=45%>