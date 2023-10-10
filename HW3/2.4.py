import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("hw3Data/emails.csv", delimiter=",")

X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

k_values = [1, 3, 5, 7, 10]

average_accuracies = []

num_folds = 5

fold_size = len(X) // num_folds
folds_X = [X[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]
folds_y = [y[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]

for k in k_values:
    accuracies = []

    for i in range(num_folds):
        X_test = folds_X[i]
        y_test = folds_y[i]
        X_train = np.concatenate(folds_X[:i] + folds_X[i+1:])
        y_train = np.concatenate(folds_y[:i] + folds_y[i+1:])

        predictions = []

        for test_point in X_test:
            distances = np.linalg.norm(X_train - test_point, axis=1)
            nearest_neighbor_indices = np.argpartition(distances, k)[:k]
            nearest_neighbor_labels = y_train[nearest_neighbor_indices]
            prediction = np.bincount(nearest_neighbor_labels).argmax()
            predictions.append(prediction)

        correct_predictions = (predictions == y_test).sum()
        accuracy = correct_predictions / len(y_test)
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    average_accuracies.append(average_accuracy)

    print(f"Average Accuracy for k={k}: {average_accuracy:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(k_values, average_accuracies, marker='o', linestyle='-')
plt.title('Average Accuracy vs. k for kNN (Manual Cross-Validation)')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Average Accuracy')
plt.grid(True)
plt.savefig("2.4.png")
plt.show()

