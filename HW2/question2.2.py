import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import graphviz
from graphviz import Source
import pygraphviz as pgv
import pandas as pd
from typing import Callable

class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, leaf=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.leaf = leaf

def entropy(labels):
    label_count = Counter(labels)
    sample_size = len(labels)
    entropy = 0.0
    for count in label_count.values():
        p = count / sample_size
        entropy -= p * np.log2(p)    
    return entropy

def information_gain_ratio(parent_labels, left_labels, right_labels):
    p = len(left_labels) / len(parent_labels)
    q = len(right_labels) / len(parent_labels)
    # conditional entropy and mutual information
    par = entropy(parent_labels)
    left = p * entropy(left_labels)
    right = q * entropy(right_labels)

    # if par == 0 or left == 0 or right == 0:
    #     # print("mutal: ", par - (left + right))
    #     return 0

    if par == 0:
        return None

    return (par - (left + right))/par

def find_best_split(data, labels):

    m, n = data.shape
    if m <= 1:
        return None, None  # No split can be made with only one sample

    best_info_gain_ratio = 0
    best_feature_idx = None
    best_threshold = None

    for feature_idx in range(n):
        unique_values = np.unique(data[:, feature_idx])
        for threshold in unique_values:

            left_mask = data[:, feature_idx] < threshold
            right_mask = ~left_mask
            left_labels = labels[left_mask]
            right_labels = labels[right_mask]

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            info_gain_ratio = information_gain_ratio(labels, left_labels, right_labels)

            if info_gain_ratio == None or info_gain_ratio == 0:
                best_feature_idx = None
                best_threshold = None
                return best_feature_idx, best_threshold
            
            if info_gain_ratio > best_info_gain_ratio:
                best_info_gain_ratio = info_gain_ratio
                best_feature_idx = feature_idx
                best_threshold = threshold
    return best_feature_idx, best_threshold

nodeCounter = 0
def build_decision_tree(data, labels):
    global nodeCounter
    nodeCounter += 1

    if len(set(labels)) == 1:
        return DecisionTreeNode(value=labels[0], leaf=True)

    best_feature_idx, best_threshold = find_best_split(data, labels)

    if best_feature_idx is None:
        return DecisionTreeNode(value=1) 

    left_mask = data[:, best_feature_idx] >= best_threshold
    right_mask = ~left_mask

    left_tree = build_decision_tree(data[left_mask], labels[left_mask])
    right_tree = build_decision_tree(data[right_mask], labels[right_mask])

    return DecisionTreeNode(feature_idx=best_feature_idx, threshold=best_threshold,
                            left=left_tree, right=right_tree)

def predict(tree, sample):
    if tree.leaf:
        return tree.value

    if sample[tree.feature_idx] >= tree.threshold:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)
    
def plot_decision_tree(root_node, filename):
    graph = pgv.AGraph(strict=True, directed=True)

    def add_node(node, parent=None):
        node_id = str(id(node))

        if node.leaf:
            label = "Class Label: " + str(node.value)
        else:
            label = "Split Feature: " +  str(node.feature_idx) + "\nSplit Threshold: " + str(node.threshold)
        graph.add_node(node_id, label=label)

        if parent is not None:
            graph.add_edge(parent, node_id)

        if node.left:
            add_node(node.left, node_id)
        if node.right:
            add_node(node.right, node_id)

    add_node(root_node)

    graph.draw(filename, format='png', prog='dot')

if __name__ == '__main__':
    # Question 2.2  
    # class_0_points = np.array([[1.0, 1.0], [1.2, 1.2], [1.4, 1.4]])
    # class_1_points = np.array([[2.0, 2.0], [2.2, 2.2], [2.4, 2.4]])
    # labels = np.array([0, 0, 0, 1, 1, 1])

    class_0_points = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    class_1_points = np.array([[0.0, 1.0]])
    labels = np.array([0, 0, 0, 1])


    # class_0_points = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # class_1_points = np.array([[1.5, 0.0]])
    # labels = np.array([0, 0, 0, 1])

    # class_0_points = np.array([[1.0, 1.0], [0.0, 0.0]])
    # class_1_points = np.array([[0.0, 1.0]])
    # labels = np.array([0, 0, 1])
#     data = {
#     "X": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
#     "Y": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
#     "Class": ["A", "B", "A", "B", "A", "B", "A", "B"]
# }

#     X = np.array(data["X"])
#     Y = np.array(data["Y"])
#     Class = np.array(data["Class"])

#     # Separate the data into class_0_points and class_1_points based on the "Class" labels
#     class_0_points = np.vstack((X[Class == "A"], Y[Class == "A"])).T
#     class_1_points = np.vstack((X[Class == "B"], Y[Class == "B"])).T

#     labels = np.array([0] * len(class_0_points) + [1] * len(class_1_points))




    training_set = np.vstack((class_0_points, class_1_points))
    # print(training_set)
    decision_tree = build_decision_tree(training_set, labels)

    plot_decision_tree(decision_tree, "2.2.tree.png")

    # Plot the training set
    plt.scatter(training_set[:, 0], training_set[:, 1], c=labels, cmap=plt.cm.Paired, marker='o')
    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.savefig('2.2.png')
    plt.show()
    