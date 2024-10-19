import numpy as np
from collections import Counter

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=str)
   
    X = data[:, :-1]
    y = data[:, -1]
   
    X_encoded = np.zeros(X.shape, dtype=float)
    for i in range(X.shape[1]):
        try:
            X_encoded[:, i] = X[:, i].astype(float)
        except ValueError:
            unique_values = np.unique(X[:, i])
            X_encoded[:, i] = np.array([np.where(unique_values == val)[0][0] for val in X[:, i]])
   
    y = (y == 'yes').astype(int) * 2 - 1
   
    return X_encoded, y

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(parent, left_child, right_child):
    weight_left = len(left_child) / len(parent)
    weight_right = len(right_child) / len(parent)
    return entropy(parent) - (weight_left * entropy(left_child) + weight_right * entropy(right_child))

def best_split(X, y):
    best_gain = -1
    best_feature, best_threshold = None, None
    
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            gain = information_gain(y, y[left_mask], y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold

def build_tree(X, y, max_depth=None, depth=0):
    if len(np.unique(y)) == 1 or (max_depth is not None and depth == max_depth):
        return Node(value=np.mean(y))
    
    feature, threshold = best_split(X, y)
    
    if feature is None:
        return Node(value=np.mean(y))
    
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    
    left_subtree = build_tree(X[left_mask], y[left_mask], max_depth, depth + 1)
    right_subtree = build_tree(X[right_mask], y[right_mask], max_depth, depth + 1)
    
    return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

def predict(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)

def bagged_trees(X, y, n_trees, max_depth=None):
    trees = []
    for _ in range(n_trees):
        indices = np.random.choice(len(X), len(X), replace=True)
        X_sample, y_sample = X[indices], y[indices]
        tree = build_tree(X_sample, y_sample, max_depth)
        trees.append(tree)
    return trees

def bagged_predict(trees, x):
    predictions = [predict(tree, x) for tree in trees]
    return np.mean(predictions)

# Load data
X_train, y_train = load_data('train_bank.csv')
X_test, y_test = load_data('test_bank.csv')

# Experiment parameters
n_repeats = 100
n_samples = 1000
n_trees = 500

single_tree_predictions = []
bagged_tree_predictions = []

for _ in range(n_repeats):
    # Sample 1,000 examples without replacement
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_sample, y_sample = X_train[indices], y_train[indices]
    
    # Train bagged trees
    trees = bagged_trees(X_sample, y_sample, n_trees)
    
    # Store predictions
    single_tree_predictions.append([predict(trees[0], x) for x in X_test])
    bagged_tree_predictions.append([bagged_predict(trees, x) for x in X_test])

single_tree_predictions = np.array(single_tree_predictions)
bagged_tree_predictions = np.array(bagged_tree_predictions)

def compute_bias_variance(predictions, true_labels):
    mean_pred = np.mean(predictions, axis=0)
    bias = np.mean((mean_pred - true_labels) ** 2)
    variance = np.mean(np.var(predictions, axis=0, ddof=1))
    return bias, variance

single_tree_bias, single_tree_variance = compute_bias_variance(single_tree_predictions, y_test)
bagged_tree_bias, bagged_tree_variance = compute_bias_variance(bagged_tree_predictions, y_test)

single_tree_error = single_tree_bias + single_tree_variance
bagged_tree_error = bagged_tree_bias + bagged_tree_variance

print("Single Tree Results:")
print(f"Bias: {single_tree_bias:.4f}")
print(f"Variance: {single_tree_variance:.4f}")
print(f"General Squared Error: {single_tree_error:.4f}")

print("\nBagged Trees Results:")
print(f"Bias: {bagged_tree_bias:.4f}")
print(f"Variance: {bagged_tree_variance:.4f}")
print(f"General Squared Error: {bagged_tree_error:.4f}")

