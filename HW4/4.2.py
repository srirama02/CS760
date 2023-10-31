import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Fetch MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist["data"].values  # Convert pandas DataFrame to numpy array
y = mnist["target"].astype(int).values  # Convert pandas Series to numpy array
X = X / 255.0  # Normalize

# One-hot encode the labels
enc = OneHotEncoder(sparse=False)
y_onehot = enc.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def softmax(z):
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def forward_propagation(X, W1, W2):
    z1 = X.dot(W1.T)
    a1 = sigmoid(z1)
    z2 = a1.dot(W2.T)
    a2 = softmax(z2)
    return z1, a1, z2, a2

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss

# Initialize parameters
input_size = 784
hidden_size = 300  # Modified as per your request
output_size = 200  # Modified as per your request
learning_rate = 0.1
epochs = 7
batch_size = 64

W1 = np.random.randn(hidden_size, input_size) * 0.01
W2 = np.random.randn(output_size, hidden_size) * 0.01

losses = []

for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward propagation
        z1, a1, z2, a2 = forward_propagation(X_batch, W1, W2)

        # Backward propagation
        dz2 = a2 - y_batch
        dW2 = np.dot(dz2.T, a1) / batch_size
        da1 = np.dot(dz2, W2)
        dz1 = da1 * sigmoid_derivative(z1)
        dW1 = np.dot(dz1.T, X_batch) / batch_size
        
        # Gradient descent step
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2

    # Compute loss at the end of the epoch
    _, _, _, a2_train = forward_propagation(X_train, W1, W2)
    loss = compute_loss(y_train, a2_train)
    losses.append(loss)

    _, _, _, a2_test = forward_propagation(X_test, W1, W2)
    predictions = np.argmax(a2_test, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    error_rate = 1 - accuracy
    
    print(f"Epoch {epoch + 1}, Test Error: {error_rate:.4f}")


plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.savefig("4.2.png")
plt.show()
