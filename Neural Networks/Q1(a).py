# Import required libraries
import numpy as np  # for numerical computations
import pandas as pd  # for data handling

# Sigmoid activation function: squashes input to range [0,1]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function, used in backpropagation
# Note: if s = sigmoid(x), then s'(x) = s(x)(1-s(x))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def initialize_weights(input_size, hidden_sizes, output_size):
    """
    Initialize weights and biases for all layers of the network
    
    Parameters:
    - input_size: number of features (4 for bank-note dataset)
    - hidden_sizes: list specifying number of neurons in each hidden layer [5,3]
    - output_size: number of output neurons (1 for binary classification)
    """
    weights = []  # list to store weight matrices
    biases = []   # list to store bias vectors
    
    # Input layer → First hidden layer
    # Shape: (hidden_sizes[0], input_size) = (5,4)
    # Each row represents weights for one neuron in first hidden layer
    weights.append(np.random.randn(hidden_sizes[0], input_size) * 0.01)
    # Shape: (hidden_sizes[0], 1) = (5,1)
    biases.append(np.zeros((hidden_sizes[0], 1)))
    
    # Between hidden layers
    for i in range(1, len(hidden_sizes)):
        # Shape: (hidden_sizes[i], hidden_sizes[i-1]) = (3,5)
        # Connects each layer to next layer
        weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i-1]) * 0.01)
        biases.append(np.zeros((hidden_sizes[i], 1)))
    
    # Last hidden layer → Output layer
    # Shape: (output_size, hidden_sizes[-1]) = (1,3)
    weights.append(np.random.randn(output_size, hidden_sizes[-1]) * 0.01)
    biases.append(np.zeros((output_size, 1)))
    
    return weights, biases

def forward_propagation(X, weights, biases):
    """
    Compute forward pass through the network
    
    Parameters:
    - X: input features for one example, shape (4,1)
    - weights: list of weight matrices
    - biases: list of bias vectors
    
    Returns:
    - activations: list of outputs from each layer
    - zs: list of weighted inputs to each layer before activation
    """
    activations = [X]  # list starts with input layer
    zs = []  # store weighted inputs before activation
    
    # For each layer's weights and biases
    for W, b in zip(weights, biases):
        # Compute weighted sum plus bias: z = Wx + b
        z = np.dot(W, activations[-1]) + b
        # Apply sigmoid activation: a = sigmoid(z)
        a = sigmoid(z)
        # Store intermediate values for backprop
        zs.append(z)
        activations.append(a)
    
    return activations, zs

def backward_propagation(X, y, weights, biases, activations, zs):
    """
    Compute gradients using backpropagation
    
    Parameters:
    - X: input features for one example
    - y: true label
    - weights, biases: network parameters
    - activations: list of outputs from forward pass
    - zs: list of weighted inputs from forward pass
    
    Returns:
    - gradients_w: list of gradients for weight matrices
    - gradients_b: list of gradients for bias vectors
    """
    gradients_w = [np.zeros_like(W) for W in weights]
    gradients_b = [np.zeros_like(b) for b in biases]
    
    # Output layer error (delta)
    # For binary cross-entropy loss with sigmoid output
    delta = activations[-1] - y.reshape(-1, 1)
    
    # Gradient for last layer
    gradients_w[-1] = np.dot(delta, activations[-2].T)
    gradients_b[-1] = delta
    
    # Backpropagate through hidden layers
    # l counts backwards through layers
    for l in range(len(weights) - 2, -1, -1):
        # delta = (next layer's weights)^T * (next layer's delta) * sigmoid'(z)
        delta = np.dot(weights[l + 1].T, delta) * sigmoid_derivative(zs[l])
        gradients_w[l] = np.dot(delta, activations[l].T)
        gradients_b[l] = delta
    
    return gradients_w, gradients_b

def load_data(train_path, test_path):
    """
    Load and prepare the bank-note dataset
    
    Parameters:
    - train_path: path to training data CSV
    - test_path: path to test data CSV
    
    Returns:
    - X_train, y_train: training features and labels
    - X_test, y_test: test features and labels
    """
    # Read CSV files
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)
    
    # Split into features (X) and labels (y)
    # .T transposes to get shape (n_features, n_examples)
    X_train = train_data.iloc[:, :-1].values.T  
    y_train = train_data.iloc[:, -1].values    
    X_test = test_data.iloc[:, :-1].values.T   
    y_test = test_data.iloc[:, -1].values      
    
    return X_train, y_train, X_test, y_test

# Example initialization
np.random.seed(2)  
input_size = 4     # bank-note has 4 features
hidden_sizes = [5, 3]  # 5 neurons in first hidden layer, 3 in second
output_size = 1    # binary classification needs 1 output neuron

# Load data
train_path = "train.csv"
test_path = "test.csv"
X_train, y_train, X_test, y_test = load_data(train_path, test_path)

# Initialize weights and biases
weights, biases = initialize_weights(input_size, hidden_sizes, output_size)

# Forward propagation
activations, zs = forward_propagation(X_train[:, [0]], weights, biases) 

# Backward propagation
gradients_w, gradients_b = backward_propagation(X_train[:, [0]], y_train[[0]], weights, biases, activations, zs)

print("Weight gradients:", gradients_w)
print("Bias gradients:", gradients_b)
print("\n")