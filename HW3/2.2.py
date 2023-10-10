import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("hw3Data/emails.csv", delimiter=",")
feature_columns = data.columns[1:-1]

predictions = []

test = data.loc[:999, feature_columns].values
training_set = data.loc[1000:4999, feature_columns].values


folds = [[0, 999], [1000, 1999], [2000, 2999], [3000, 3999], [4000, 4999]]
for fold in folds:
    test = data.loc[fold[0]:fold[1], feature_columns].values

    training_set = np.concatenate([
        data.loc[0:fold[0]-1, feature_columns].values,
        data.loc[fold[1]+1:, feature_columns].values
    ])

    temp12 = np.concatenate([
        data.loc[0:fold[0]-1, "Prediction"].values,
        data.loc[fold[1]+1:, "Prediction"].values
    ])

    predictions = []
    for test_point in test:
        distances = np.linalg.norm(training_set - test_point, axis=1)
        nearest_neighbor_index = np.argmin(distances)
        nearest_neighbor_label = temp12[nearest_neighbor_index]
        predictions.append(nearest_neighbor_label)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and data.iloc[fold[0] + i]["Prediction"] == 1:
            TP += 1
        elif predictions[i] == 1 and data.iloc[fold[0] + i]["Prediction"] == 0:
            FP += 1
        elif predictions[i] == 0 and data.iloc[fold[0] + i]["Prediction"] == 1:
            FN += 1
        else:
            TN += 1
    accuracy = (TP + TN) / len(predictions)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall)