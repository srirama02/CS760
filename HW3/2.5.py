import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_curve,roc_auc_score

data = pd.read_csv('hw3Data/emails.csv')

X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

X_train = X[:4000]
y_train = y[:4000]

X_test = X[4000:]
y_test = y[4000:]


def k_nearest_neighbors(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = np.linalg.norm(X_train - test_point, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        label_counts = Counter(nearest_labels)
        most_common_label = label_counts.most_common(1)[0][0]
        confidence_rate = label_counts[1]/k

        predictions.append(confidence_rate)
    return np.array(predictions)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(X_train,y_train,X_test,learning_rate, num_iterations):
    num_samples, num_features = X_train.shape
    weights = np.zeros(num_features)
    
    for _ in range(num_iterations):
        linear_model = np.dot(X_train, weights)
        predictions = sigmoid(linear_model)
        
        gradient = np.dot(X_train.T, (predictions - y_train)) 
        
        weights -= learning_rate * gradient
    
    linear_model = np.dot(X_test, weights)
    predictions = sigmoid(linear_model)
    
    return predictions

k = 5
predicted_labels = k_nearest_neighbors(X_train, y_train, X_test, k)

fpr_knn, tpr_knn, thresholds = roc_curve(y_test, predicted_labels)
roc_auc_knn = roc_auc_score(y_test, predicted_labels)


learning_rate = 0.0001
num_iterations = 2000
predicted_labels = logistic_regression(X_train,y_train,X_test,learning_rate, num_iterations)
fpr_logistic, tpr_logistic, thresholds = roc_curve(y_test, predicted_labels)
roc_auc_logistic = roc_auc_score(y_test,predicted_labels)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label=f'k-NN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_logistic, tpr_logistic, color='red', lw=2, label=f'Logistic Regression (AUC = {roc_auc_logistic:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('2.5.png')
plt.show()
