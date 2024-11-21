def evaluate_model(model, X_test, Y_test):
    predictions = model.predict(X_test)
    # top_ten = sorted(predictions, reverse=True)[:10]
    # print("Top ten predictions:", top_ten)
    predictions = [1 if x >= 0.5 else 0 for x in predictions]
    TP = sum((predictions[i] == 1) and (Y_test[i] == 1) for i in range(len(Y_test)))
    TN = sum((predictions[i] == 0) and (Y_test[i] == 0) for i in range(len(Y_test)))
    FP = sum((predictions[i] == 1) and (Y_test[i] == 0) for i in range(len(Y_test)))
    FN = sum((predictions[i] == 0) and (Y_test[i] == 1) for i in range(len(Y_test)))
    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP)!= 0 else 0
    recall = TP / (TP + FN) if (TP + FN)!= 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall)!= 0 else 0
    print("Accuracy:", accuracy, "Precision:", precision, "Recall:", recall, "F1 Score:", f1_score)
    
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_results(model, X_test, Y_test):    
    X_plot = X_test[:, -3:]
    Y_test = Y_test.ravel()

    predictions = model.predict(X_test)
    predictions = (predictions >= 0.5).astype(int)
    
    # Define classes for TP, FP, TN, FN
    TP_mask = (predictions == 1) & (Y_test == 1)
    FP_mask = (predictions == 1) & (Y_test == 0)
    TN_mask = (predictions == 0) & (Y_test == 0)
    FN_mask = (predictions == 0) & (Y_test == 1)
    
    # Plot the data
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_plot[TP_mask][:, 0], X_plot[TP_mask][:, 1], X_plot[TP_mask][:, 2], 
               c='green', label='True Positive (TP)', alpha=0.8, edgecolor='k', marker='o')
    ax.scatter(X_plot[FP_mask][:, 0], X_plot[FP_mask][:, 1], X_plot[FP_mask][:, 2], 
               c='red', label='False Positive (FP)', alpha=0.8, edgecolor='k', marker='x')
    ax.scatter(X_plot[TN_mask][:, 0], X_plot[TN_mask][:, 1], X_plot[TN_mask][:, 2], 
               c='blue', label='True Negative (TN)', alpha=0.8, edgecolor='k', marker='^')
    ax.scatter(X_plot[FN_mask][:, 0], X_plot[FN_mask][:, 1], X_plot[FN_mask][:, 2], 
               c='orange', label='False Negative (FN)', alpha=0.8, edgecolor='k', marker='s')
    ax.set_title('Classification Results with Confusion Components')
    ax.legend()
    plt.savefig('confusion_results.png')
    plt.show()
