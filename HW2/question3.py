import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = np.loadtxt('data/Dbig.txt', delimiter=' ')

# rand_ind_data = np.random.permutation(dataset.shape[0])
# dataset = dataset[rand_ind_data, :]
np.random.shuffle(dataset)

candidate_training_set = dataset[:8192]
test_set = dataset[8192:]

training_set_sizes = [32, 128, 512, 2048, 8192]

node_counts = []
test_errors = []

for n in training_set_sizes:
    current_training_set = candidate_training_set[:n]
    current_test_set = test_set
    
    X_train = current_training_set[:, :-1]
    y_train = current_training_set[:, -1]
    
    X_test = current_test_set[:, :-1]
    y_test = current_test_set[:, -1]
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_pred)
    
    node_counts.append(clf.tree_.node_count)
    test_errors.append(test_error)

for i in range(len(training_set_sizes)):
    print(f"n = {training_set_sizes[i]}, Number of Nodes = {node_counts[i]}, Error = {test_errors[i]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(training_set_sizes, test_errors, marker='o', linestyle='-')
plt.title("n vs. errn")
plt.xlabel("Training Set Size (n)")
plt.ylabel("Test Set Error")
plt.grid(True)
plt.savefig("3.plot.png")
# plt.show()
