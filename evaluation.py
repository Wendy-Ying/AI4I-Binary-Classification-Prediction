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