import numpy as np
from scipy.optimize import minimize
from typing import Tuple
import time

class GaussianSVM:
    def __init__(self, C: float, gamma: float):
        """
        Initialize Gaussian kernel SVM.
        
        Args:
            C: Regularization parameter
            gamma: Gaussian kernel parameter
        """
        self.C = C
        self.gamma = gamma
        self.alphas = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian kernel between X1 and X2.
        K(x,y) = exp(-γ||x-y||²)
        """
        # Compute pairwise squared Euclidean distances
        # Using matrix operation: ||x-y||² = ||x||² + ||y||² - 2x·y
        norm1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        norm2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances = norm1 + norm2 - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * distances)
    
    def _objective(self, alphas: np.ndarray, K: np.ndarray, y: np.ndarray) -> float:
        """Compute the dual objective function."""
        return -np.sum(alphas) + 0.5 * np.sum(y.reshape(-1,1) * y * K * alphas.reshape(-1,1) * alphas)
    
    def _objective_gradient(self, alphas: np.ndarray, K: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of the dual objective."""
        return -np.ones_like(alphas) + (y.reshape(-1,1) * y * K * alphas.reshape(-1,1)).sum(axis=1)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM using dual formulation with Gaussian kernel."""
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda a: np.dot(a, y)},
        ]
        
        # Box constraints
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Initialize alphas
        alpha0 = np.zeros(n_samples)
        
        # Solve dual optimization problem
        result = minimize(
            fun=lambda a: self._objective(a, K, y),
            x0=alpha0,
            method='SLSQP',
            jac=lambda a: self._objective_gradient(a, K, y),
            bounds=bounds,
            constraints=constraints
        )
        
        self.alphas = result.x
        
        # Find support vectors (alphas > 1e-3)
        sv_indices = np.where(self.alphas > 1e-3)[0]
        self.support_vector_indices = sv_indices
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = self.alphas[sv_indices]
        
        # Compute bias
        kernel_sv = self._compute_kernel(self.support_vectors, self.support_vectors)
        self.b = np.mean(
            self.support_vector_labels - 
            (self.support_vector_alphas * self.support_vector_labels).dot(kernel_sv)
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using kernel trick."""
        kernel = self._compute_kernel(X, self.support_vectors)
        return np.sign(
            np.sum(
                self.support_vector_alphas * self.support_vector_labels * kernel, 
                axis=1
            ) + self.b
        )
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return np.mean(self.predict(X) == y)

def load_and_preprocess_data(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the banknote data."""
    train_data = np.loadtxt(train_path, delimiter=',')
    test_data = np.loadtxt(test_path, delimiter=',')
    
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_and_preprocess_data('train.csv', 'test.csv')
    
    # Parameters to test
    C_values = [100/873, 500/873, 700/873]
    gamma_values = [0.1, 0.5, 1, 5, 100]
    
    # Store results
    results = []
    
    print("\nTraining Gaussian Kernel SVM:")
    print("\nC\tγ\tTrain Error\tTest Error\t#Support Vectors\tTime(s)")
    print("-" * 75)
    
    for C in C_values:
        for gamma in gamma_values:
            start_time = time.time()
            
            # Train model
            svm = GaussianSVM(C=C, gamma=gamma)
            svm.fit(X_train, y_train)
            
            # Calculate errors
            train_error = 1 - svm.score(X_train, y_train)
            test_error = 1 - svm.score(X_test, y_test)
            n_support = len(svm.support_vector_indices)
            
            training_time = time.time() - start_time
            
            # Store results
            results.append({
                'C': C,
                'gamma': gamma,
                'train_error': train_error,
                'test_error': test_error,
                'n_support': n_support,
                'time': training_time
            })
            
            print(f"{C:.4f}\t{gamma:.1f}\t{train_error:.4f}\t\t{test_error:.4f}\t\t{n_support}\t\t{training_time:.2f}")
    
    # Find best combination based on test error
    best_result = min(results, key=lambda x: x['test_error'])
    print("\nBest combination:")
    print(f"C = {best_result['C']:.4f}")
    print(f"γ = {best_result['gamma']:.1f}")
    print(f"Training error: {best_result['train_error']:.4f}")
    print(f"Test error: {best_result['test_error']:.4f}")
    print(f"Number of support vectors: {best_result['n_support']}")