import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz


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

            if info_gain_ratio == None:
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

if __name__ == '__main__':
    data = np.loadtxt('data/D1.txt', delimiter=' ')
    # X = data[:, :-1]  # Features x1 and x2
    # y = data[:, -1]   # Binary class labels (0 or 1)

    X = data[:600, :-1]  # Features x1 and x2
    y = data[:600, -1]   # Binary class labels (0 or 1)

    # Build the decision tree
    decision_tree = build_decision_tree(X, y)

    # Make predictions on new data
    # Replace 'new_data' with your test data
    # new_data = np.array([[3.0, 4.0], [1.0, 2.0]])

    new_data = data[600:, :-1]  # Features x1 and x2
    correct_labels = data[600:, -1]   # Binary class labels (0 or 1)
    predictions = [predict(decision_tree, sample) for sample in new_data]

    count = 0
    for i in range(len(predictions)):
        if correct_labels[i] != predictions[i]:
            count += 1
    
    print("error rate: ", (count / len(predictions))*100)