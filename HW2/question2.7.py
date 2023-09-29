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
    
# Added from post on Piazza
def draw_decision_boundary(model_function:Callable, grid_abs_bound:float=1.0,savefile:str=None):
    colors=['#91678f','#afd6d2'] # hex color for [y=0, y=1]

    xval = np.linspace(grid_abs_bound,-grid_abs_bound,1000).tolist() # grid generation
    xdata = []
    for i in range(len(xval)):
        for j in range(len(xval)):
            xdata.append([xval[i],xval[j]])

    df = pd.DataFrame(data=xdata,columns=['x_1','x_2']) # creates a dataframe to standardize labels
    df['y'] = df.apply(model_function,axis=1) # applies model from model_function arg
    d_columns = df.columns.to_list() # grabs column headers
    y_label = d_columns[-1] # uses last header as label
    d_xfeature = d_columns[0] # uses first header as x_1 feature
    d_yfeature = d_columns[1] # uses second header as x_1 feature
    df = df.sort_values(by=y_label) # sorts by label to ensure correct ordering in plotting loop

    # plt.figure(figsize=(10, 10)) 

    d_xlabel = f"feature  $\mathit{{{d_xfeature}}}$" # label for x-axis
    dy_ylabel = f"feature  $\mathit{{{d_yfeature}}}$" # label for y-axis
    plt.xlabel(d_xlabel, fontsize=10) # set x-axis label
    plt.ylabel(dy_ylabel, fontsize=10) # set y-axis label
    legend_labels = [] # create container for legend labels to ensure correct ordering

    for i,label in enumerate(df[y_label].unique().tolist()): # loop through placeholder dataframe
        df_set = df[df[y_label]==label] # sort according to label
        set_x = df_set[d_xfeature] # grab x_1 feature set
        set_y = df_set[d_yfeature] # grab x_2 feature set
        plt.scatter(set_x,set_y,c=colors[i],marker='s', s=40) # marker='s' for square, s=40 for size of squares large enough
        legend_labels.append(f"""{y_label} = {label}""") # apply labels for legend in the same order as sorted dataframe

    plt.title("Model Decision Boundary", fontsize=12) # set plot title
    ax = plt.gca() # grab to set background color of plot
    ax.set_facecolor('#2b2d2e') # set aforementioned background color in hex color
    # plt.legend(legend_labels) # create legend with sorted labels
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    

    if savefile is not None: # save your plot as .png file
        plt.savefig(savefile)

if __name__ == '__main__':

    data = np.loadtxt('data/Dbig.txt', delimiter=' ')

    np.random.shuffle(data)

    candidate_training_set = data[:8192]
    test_set = data[8192:]

    training_set_sizes = [32, 128, 512, 2048, 8192]

    node_counts = []
    error_rates = []

    for n in training_set_sizes:
        current_training_set = candidate_training_set[:n]
        current_test_set = test_set  
        X_train = current_training_set[:, :-1]
        y_train = current_training_set[:, -1]
        
        X_test = current_test_set[:, :-1]
        y_test = current_test_set[:, -1]
        nodeCounter = 0
        decision_tree = build_decision_tree(X_train, y_train)

        predictions = [predict(decision_tree, sample) for sample in X_test]
        

        count = 0
        for i in range(len(predictions)):
            if y_test[i] != predictions[i]:
                count += 1
        errorRate = (count / len(predictions))
        node_counts.append(nodeCounter)
        error_rates.append(errorRate)
        print("D_n: ", n, " number of nodes: ", nodeCounter, " error rate: ", errorRate)

        def model(row):
            return predict(decision_tree, row)
        
        draw_decision_boundary(model_function=model, grid_abs_bound=5, savefile="2.7." + str(n) + ".db.png")



    plt.figure(figsize=(8, 6))
    plt.plot(node_counts, error_rates, marker='o', linestyle='-', color='b')
    plt.title('Number of Nodes vs. Error Rate')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Error Rate')
    plt.grid(True)
    plt.savefig("2.7.plot.png")