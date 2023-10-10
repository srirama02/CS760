import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("hw3Data/D2z.txt", delimiter=" ", header=None, names=['Feature1', 'Feature2', 'Label'])

training_set = data[["Feature1", "Feature2"]].values

x_range = np.arange(-2, 2.1, 0.1)
y_range = np.arange(-2, 2.1, 0.1)
xx, yy = np.meshgrid(x_range, y_range)
test_points = np.c_[xx.ravel(), yy.ravel()]
print(test_points)
predictions = []

for test_point in test_points:
    distances = np.linalg.norm(training_set - test_point, axis=1)
    nearest_neighbor_index = np.argmin(distances)
    nearest_neighbor_label = data.iloc[nearest_neighbor_index]["Label"]
    predictions.append(nearest_neighbor_label)

predictions = np.array(predictions)

plt.scatter(data[data["Label"] == 0]["Feature1"], data[data["Label"] == 0]["Feature2"], facecolors='none', edgecolors='black', label='Training Set (Class 0)')
plt.scatter(data[data["Label"] == 1]["Feature1"], data[data["Label"] == 1]["Feature2"], c='black', marker='+', facecolors='none', label='Training Set (Class 1)')

plt.scatter(test_points[:, 0], test_points[:, 1], c=predictions, marker='.', cmap=plt.cm.coolwarm, label='Predictions')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('1NN Predictions on 2D Grid')
plt.grid(True)
plt.savefig("2.1.png")
plt.show()

