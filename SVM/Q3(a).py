import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict

class DualSVM:
    def __init__(self, C: float):
        """
        Initialize Dual SVM.
        
        Args:
            C: Regularization parameter
        """
        self.C = C
        self.alphas = None  # Lagrange multipliers
        self.w = None       # Weight vector
        self.b = 0         # Bias term
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute linear kernel between X1 and X2."""
        return np.dot(X1, X2.T)
    
    def _objective(self, alphas: np.ndarray, K: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the dual objective function.
        
        Args:
            alphas: Lagrange multipliers
            K: Kernel matrix
            y: Labels
        
        Returns:
            Objective value
        """
        return -np.sum(alphas) + 0.5 * np.sum(y.reshape(-1,1) * y * K * alphas.reshape(-1,1) * alphas)
    
    def _objective_gradient(self, alphas: np.ndarray, K: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of the dual objective."""
        return -np.ones_like(alphas) + (y.reshape(-1,1) * y * K * alphas.reshape(-1,1)).sum(axis=1)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM using dual formulation.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels {-1, 1} (n_samples,)
        """
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        
        # Define constraints for optimization
        constraints = [
            {'type': 'eq', 'fun': lambda a: np.dot(a, y)},  # Sum(alpha_i * y_i) = 0
        ]
        
        # Box constraints: 0 <= alpha_i <= C
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
        
        # Find support vectors (alphas > 1e-5)
        sv_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vector_indices = sv_indices
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        # Compute w and b
        self.w = np.sum(self.alphas.reshape(-1, 1) * y.reshape(-1, 1) * X, axis=0)
        
        # Compute b using support vectors
        margins = np.dot(self.support_vectors, self.w)
        self.b = np.mean(self.support_vector_labels - margins)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for test data."""
        return np.sign(np.dot(X, self.w) + self.b)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return np.mean(self.predict(X) == y)

def load_and_preprocess_data(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the banknote data."""
    train_data = np.loadtxt(train_path, delimiter=',')
    test_data = np.loadtxt(test_path, delimiter=',')
    
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    
    # Convert labels to {-1, 1}
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1
    
    return X_train, y_train, X_test, y_test

# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data('train.csv', 'test.csv')
    
    # Define C values to test
    C_values = [100/873, 500/873, 700/873]
    
    print("Training Dual SVM and comparing with Primal SVM results:")
    print("\nC\tTrain Error\tTest Error\t#Support Vectors")
    print("-" * 60)
    
    for C in C_values:
        # Train dual SVM
        svm = DualSVM(C=C)
        svm.fit(X_train, y_train)
        
        # Calculate errors
        train_error = 1 - svm.score(X_train, y_train)
        test_error = 1 - svm.score(X_test, y_test)
        n_support = len(svm.support_vector_indices)
        
        print(f"{C:.4f}\t{train_error:.4f}\t\t{test_error:.4f}\t\t{n_support}")
        
        # Print weight vector for comparison with primal SVM
        print(f"\nWeight vector norm: {np.linalg.norm(svm.w):.4f}")
        print(f"Bias term: {svm.b:.4f}")