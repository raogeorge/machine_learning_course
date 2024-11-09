import numpy as np

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    
    X = data[:, :-1]
    y = data[:, -1]
    
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    y = 2 * y - 1
    
    return X_with_bias, y

def voted_perceptron(X, y, T=10):
    n_samples, n_features = X.shape
    
    w_list = []
    c_list = []
    
    w = np.zeros(n_features)
    c = 0
    
    for epoch in range(T):
        for i in range(n_samples):
            prediction = np.sign(np.dot(w, X[i]))
            
            if prediction * y[i] <= 0: 
                if c > 0:
                    w_list.append(w.copy())
                    c_list.append(c)
                
                w = w + y[i] * X[i]
                c = 1
            else:
                c += 1 
                
        print(f"Epoch {epoch + 1}, distinct weight vectors: {len(w_list)}")
    
    if c > 0:
        w_list.append(w)
        c_list.append(c)
    
    return np.array(w_list), np.array(c_list)

def predict_voted(X, w_list, c_list):
    predictions = np.sign(np.dot(X, w_list.T))  
    
    weighted_predictions = predictions * c_list
    
    final_predictions = np.sign(weighted_predictions.sum(axis=1))
    
    return final_predictions

def evaluate(y_true, y_pred):
    return np.mean(y_true != y_pred)

print("Loading data...")
train_data, train_labels = load_data('train.csv')
test_data, test_labels = load_data('test.csv')

print("\nTraining Voted Perceptron...")
w_list, c_list = voted_perceptron(train_data, train_labels, T=10)

train_pred = predict_voted(train_data, w_list, c_list)
train_error = evaluate(train_labels, train_pred)
train_correct = sum(train_pred == train_labels)

test_pred = predict_voted(test_data, w_list, c_list)
test_error = evaluate(test_labels, test_pred)

print("\nResults:")
print(f"Number of distinct weight vectors: {len(w_list)}")
print(f"Number of correctly predicted training examples: {train_correct}")
print(f"Training error rate: {train_error:.4f}")
print(f"Test error rate: {test_error:.4f}")

print("\nWeight vectors and their counts:")
for i, (w, c) in enumerate(zip(w_list, c_list)):
    print(f"\nWeight vector {i+1}:")
    print(f"Count (survival time): {c}")
    print("Weights by feature:")
    feature_names = ['bias', 'variance', 'skewness', 'curtosis', 'entropy']
    for name, weight in zip(feature_names, w):
        print(f"{name}: {weight:.4f}")