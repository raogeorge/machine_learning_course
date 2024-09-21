import math
from collections import Counter

class Node:
    def __init__(self, attribute=None, label=None, branches=None):
        self.attribute = attribute
        self.label = label
        self.branches = branches or {}

def id3(data, attributes, label_attribute, max_depth, criterion='info_gain'):
    labels = [row[label_attribute] for row in data]
    
    if len(set(labels)) == 1:
        return Node(label=labels[0])
    if len(attributes) == 0 or max_depth == 0:
        return Node(label=max(set(labels), key=labels.count))
    
    best_attribute = choose_best_attribute(data, attributes, label_attribute, criterion)
    node = Node(attribute=best_attribute)
    
    for value in set(row[best_attribute] for row in data):
        subset = [row for row in data if row[best_attribute] == value]
        if len(subset) == 0:
            node.branches[value] = Node(label=max(set(labels), key=labels.count))
        else:
            remaining_attributes = [attr for attr in attributes if attr != best_attribute]
            node.branches[value] = id3(subset, remaining_attributes, label_attribute, max_depth - 1, criterion)
    
    return node

def choose_best_attribute(data, attributes, label_attribute, criterion):
    if criterion == 'info_gain':
        return max(attributes, key=lambda attr: information_gain(data, attr, label_attribute))
    elif criterion == 'majority_error':
        return max(attributes, key=lambda attr: majority_error(data, attr, label_attribute))
    elif criterion == 'gini_index':
        return max(attributes, key=lambda attr: gini_index(data, attr, label_attribute))
    else:
        raise ValueError("Invalid criterion. Choose 'info_gain', 'majority_error', or 'gini_index'.")

def entropy(data, attribute):
    values = [row[attribute] for row in data]
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def information_gain(data, attribute, label_attribute):
    total_entropy = entropy(data, label_attribute)
    weighted_entropy = 0
    for value in set(row[attribute] for row in data):
        subset = [row for row in data if row[attribute] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset, label_attribute)
    return total_entropy - weighted_entropy

def majority_error(data, attribute, label_attribute):
    total_error = 1 - max(Counter(row[label_attribute] for row in data).values()) / len(data)
    weighted_error = 0
    for value in set(row[attribute] for row in data):
        subset = [row for row in data if row[attribute] == value]
        weight = len(subset) / len(data)
        majority_class = max(set(row[label_attribute] for row in subset), key=lambda c: sum(1 for row in subset if row[label_attribute] == c))
        error = sum(1 for row in subset if row[label_attribute] != majority_class) / len(subset)
        weighted_error += weight * error
    return total_error - weighted_error

def gini_index(data, attribute, label_attribute):
    total_gini = 1 - sum((count / len(data)) ** 2 for count in Counter(row[label_attribute] for row in data).values())
    weighted_gini = 0
    for value in set(row[attribute] for row in data):
        subset = [row for row in data if row[attribute] == value]
        weight = len(subset) / len(data)
        gini = 1 - sum((count / len(subset)) ** 2 for count in Counter(row[label_attribute] for row in subset).values())
        weighted_gini += weight * gini
    return total_gini - weighted_gini

def predict(node, instance):
    if node.label is not None:
        return node.label
    value = instance[node.attribute]
    if value not in node.branches:
        return max(node.branches.values(), key=lambda n: n.label if n.label else '')
    return predict(node.branches[value], instance)

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            data.append({
                'buying': values[0],
                'maint': values[1],
                'doors': values[2],
                'persons': values[3],
                'lug_boot': values[4],
                'safety': values[5],
                'label': values[6]
            })
    return data

def calculate_error(tree, data):
    incorrect = sum(1 for instance in data if predict(tree, instance) != instance['label'])
    return incorrect / len(data)

def run_experiment(train_data, test_data, max_depths, criteria):
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    results = {criterion: {depth: {'train': 0, 'test': 0} for depth in max_depths} for criterion in criteria}
    
    for criterion in criteria:
        for depth in max_depths:
            tree = id3(train_data, attributes, 'label', depth, criterion)
            train_error = calculate_error(tree, train_data)
            test_error = calculate_error(tree, test_data)
            results[criterion][depth]['train'] = train_error
            results[criterion][depth]['test'] = test_error
    
    return results

train_data = load_data('train.csv')
test_data = load_data('test.csv')

max_depths = range(1, 7)
criteria = ['info_gain', 'majority_error', 'gini_index']
results = run_experiment(train_data, test_data, max_depths, criteria)

print("Depth | Information Gain | Majority Error | Gini Index")
print("      | Train  | Test    | Train  | Test   | Train | Test")
print("------|--------|---------|--------|--------|-------|------")
for depth in max_depths:
    print(f"{depth:5d} | {results['info_gain'][depth]['train']:.4f} | {results['info_gain'][depth]['test']:.4f} | "
          f"{results['majority_error'][depth]['train']:.4f} | {results['majority_error'][depth]['test']:.4f} | "
          f"{results['gini_index'][depth]['train']:.4f} | {results['gini_index'][depth]['test']:.4f}")
