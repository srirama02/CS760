import numpy as np
import pandas as pd

data = pd.read_csv("hw3Data/emails.csv", delimiter=",")

y = data['Prediction'].to_numpy()  # Convert to NumPy array

X = data.drop('Prediction', axis=1).values[:,1:].astype(float)
y = data['Prediction'].astype(float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, h - y)
        theta -= learning_rate * gradient
    
    return theta

def logistic_regression(X_train, y_train, X_test, y_test, learning_rate, num_iterations):
    theta = gradient_descent(X_train, y_train, learning_rate, num_iterations)
    
    z = np.dot(X_test, theta)
    y_pred = sigmoid(z)
    
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    correct_predictions = (y_pred_binary == y_test).astype(int)
    accuracy = np.mean(correct_predictions)
    
    true_positives = np.sum((y_pred_binary == 1) & (y_test == 1))
    false_positives = np.sum((y_pred_binary == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred_binary == 0) & (y_test == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return accuracy, precision, recall

num_folds = 5
fold_size = len(X) // num_folds

accuracies = []
precisions = []
recalls = []

for fold in range(num_folds):
    start_idx = fold * fold_size
    end_idx = (fold + 1) * fold_size
    X_train = np.concatenate([X[:start_idx], X[end_idx:]])
    y_train = np.concatenate([y[:start_idx], y[end_idx:]])
    
    X_test = X[start_idx:end_idx]
    y_test = y[start_idx:end_idx]
    
    learning_rate = 0.01
    num_iterations = 1000
    accuracy, precision, recall = logistic_regression(X_train, y_train, X_test, y_test, learning_rate, num_iterations)
    
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

for fold in range(num_folds):
    print(f"Fold {fold + 1}:")
    print(f"Accuracy: {accuracies[fold]}")
    print(f"Precision: {precisions[fold]}")
    print(f"Recall: {recalls[fold]}")
    print()

avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)

print("Average Accuracy:", avg_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
