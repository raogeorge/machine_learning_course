import numpy as np

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

def perceptron(X, y, T=10):
    """
    Standard Perceptron implementation
    X: features with bias term (n_samples, n_features + 1)
    y: labels (-1 or 1)
    T: maximum number of epochs
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features) 
    
    for epoch in range(T):
        mistakes = 0
        for i in range(n_samples):
            prediction = np.sign(np.dot(w, X[i]))
            if prediction * y[i] <= 0:  
                w += y[i] * X[i]  
                mistakes += 1
        
        print(f"Epoch {epoch + 1}, mistakes: {mistakes}")
        if mistakes == 0: 
            print(f"Converged at epoch {epoch + 1}")
            break
    
    return w

def evaluate(X, y, w):
    """Calculate prediction error"""
    predictions = np.sign(np.dot(X, w))
    errors = np.sum(predictions != y)
    return errors / len(y)

train_data, train_labels = load_data('train.csv')

print("Training Perceptron...")
w = perceptron(train_data, train_labels, T=10)

test_data, test_labels = load_data('test.csv')
test_error = evaluate(test_data, test_labels, w)

print("\nFinal weight vector:", w)
print(f"Test error rate: {test_error:.4f}")

feature_names = ['bias', 'variance', 'skewness', 'curtosis', 'entropy']
print("\nWeights by feature:")
for name, weight in zip(feature_names, w):
    print(f"{name}: {weight:.4f}")
