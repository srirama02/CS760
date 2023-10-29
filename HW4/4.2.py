import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target.astype(int)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data: feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target labels
num_classes = 10
y_train_onehot = np.zeros((y_train.shape[0], num_classes))
y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

# Define hyperparameters
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.01  # Reduce the learning rate
num_epochs = 10  # Define the number of training epochs


# Initialize weights and biases
np.random.seed(0)  # Set a fixed random seed for reproducibility
W1 = np.random.randn(hidden_size, input_size)
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size)
b2 = np.zeros((output_size, 1))

# Define sigmoid and softmax functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    for i in range(X_train.shape[0]):
        # Forward pass
        x = X_train[i].reshape(-1, 1)
        z1 = np.dot(W1, x) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = softmax(z2)

        # Compute the cross-entropy loss
        loss = -np.sum(y_train_onehot[i] * np.log(a2))
        total_loss += loss

        # Backpropagation
        d2 = a2 - y_train_onehot[i].reshape(-1, 1)
        dW2 = np.dot(d2, a1.T)
        db2 = d2
        d1 = np.dot(W2.T, d2) * a1 * (1 - a1)
        dW1 = np.dot(d1, x.T)
        db1 = d1

        # Update weights and biases using gradient descent
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Calculate average loss for the epoch
    avg_loss = total_loss / X_train.shape[0]
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Test the model on the test set
num_correct = 0
for i in range(X_test.shape[0]):
    x = X_test[i].reshape(-1, 1)
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)
    predicted_label = np.argmax(a2)

    if predicted_label == y_test[i]:
        num_correct += 1

accuracy = num_correct / X_test.shape[0]
print(f"Test Accuracy: {accuracy * 100:.2f}%")
