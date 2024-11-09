import numpy as np

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    
    X = data[:, :-1]
    y = data[:, -1]
    
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    y = 2 * y - 1
    
    return X_with_bias, y

def average_perceptron(X, y, T=10):
    n_samples, n_features = X.shape
    
    w = np.zeros(n_features)
    a = np.zeros(n_features)
    
    count = 1
    
    for epoch in range(T):
        mistakes = 0
        for i in range(n_samples):
            prediction = np.sign(np.dot(w, X[i]))
            
            if prediction * y[i] <= 0: 
                mistakes += 1
                w = w + y[i] * X[i]
            
            a = a + w
            count += 1
                
        print(f"Epoch {epoch + 1}, mistakes: {mistakes}")
    
    a = a / count
    
    return a

def predict(X, w):
    return np.sign(np.dot(X, w))

def evaluate(y_true, y_pred):
    return np.mean(y_true != y_pred)

print("Loading data...")
train_data, train_labels = load_data('train.csv')
test_data, test_labels = load_data('test.csv')

print("\nTraining Average Perceptron...")
w_avg = average_perceptron(train_data, train_labels, T=10)

train_pred = predict(train_data, w_avg)
test_pred = predict(test_data, w_avg)

train_error = evaluate(train_labels, train_pred)
test_error = evaluate(test_labels, test_pred)

print("\nResults:")
print("Average weight vector:")
feature_names = ['bias', 'variance', 'skewness', 'curtosis', 'entropy']
for name, weight in zip(feature_names, w_avg):
    print(f"{name}: {weight:.4f}")

print(f"\nTraining error rate: {train_error:.4f}")
print(f"Test error rate: {test_error:.4f}")