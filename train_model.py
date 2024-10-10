import numpy as np

# Define RDAModel class directly in the file
class RDAModel:
    def __init__(self, alpha=0.5):
        self.alpha = alpha  # Regularization parameter
        self.means = None
        self.cov_matrix = None
        self.priors = None
        self.classes = None

    def fit(self, X, y):
        # Identify the unique classes
        self.classes = np.unique(y)
        n_features = X.shape[1]

        # Initialize means and covariance matrix for each class
        self.means = {cls: np.mean(X[y == cls], axis=0) for cls in self.classes}
        cov_matrices = {cls: np.cov(X[y == cls].T) for cls in self.classes}
        
        # Calculate the pooled covariance matrix
        pooled_cov = sum(cov_matrices[cls] * (np.sum(y == cls) - 1) for cls in self.classes) / (len(y) - len(self.classes))
        self.cov_matrix = self.alpha * pooled_cov + (1 - self.alpha) * np.identity(n_features)
        
        # Calculate priors
        self.priors = {cls: np.mean(y == cls) for cls in self.classes}

        return self

    def predict(self, X):
        predictions = []
        for x in X:
            discriminants = {}
            for cls in self.classes:
                mean_diff = x - self.means[cls]
                inv_cov = np.linalg.inv(self.cov_matrix)
                discriminants[cls] = -0.5 * np.dot(np.dot(mean_diff, inv_cov), mean_diff.T) + np.log(self.priors[cls])
            predictions.append(max(discriminants, key=discriminants.get))
        return np.array(predictions)

# Function to generate synthetic data for K-class classification
def generate_random_data(n_samples_per_class, n_features, n_classes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = []
    y = []
    
    # Generate data for each class
    for class_label in range(n_classes):
        # Center each class around a different point in feature space
        class_center = np.random.uniform(-5, 5, n_features)  
        class_data = np.random.randn(n_samples_per_class, n_features) + class_center
        X.append(class_data)
        y.extend([class_label] * n_samples_per_class)
    
    X = np.vstack(X)  # Stack vertically to form a matrix
    y = np.array(y)   # Convert list to numpy array
    
    return X, y

# Training function
def train_model(n_samples_per_class, n_features, n_classes, alpha=0.5):
    # Create an instance of the RDAModel
    rda_model = RDAModel(alpha=alpha)

    # Generate synthetic data for K-class classification
    X, y = generate_random_data(n_samples_per_class=n_samples_per_class, n_features=n_features, n_classes=n_classes)

    # Train the model
    rda_model.fit(X, y)

    print("Model training completed.")
    return rda_model

if __name__ == "__main__":
    # Example usage: specify parameters
    n_samples_per_class = 100
    n_features = 6
    n_classes = 5
    alpha = 0.5

    train_model(n_samples_per_class=n_samples_per_class, n_features=n_features, n_classes=n_classes, alpha=alpha)
