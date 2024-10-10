import numpy as np
import matplotlib.pyplot as plt
import argparse
from train_model import RDAModel, generate_random_data
import pandas as pd

def test_rda_model(n_samples_per_class, n_features, n_classes, alpha=0.5, seed=None):
    # Generate synthetic data with a specified seed for reproducibility
    X, y = generate_random_data(n_samples_per_class=n_samples_per_class, n_features=n_features, n_classes=n_classes, seed=seed)

    # Create and train the RDA model
    rda_model = RDAModel(alpha=alpha)
    rda_model.fit(X, y)

    # Predict the classes for the synthetic data
    y_pred = rda_model.predict(X)

    # Plot the original data with true labels (for the first two features)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"True Classes (n_classes={n_classes})")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(*scatter.legend_elements(), title="Classes")

    # Plot the data with predicted labels
    plt.subplot(1, 2, 2)
    plt.title(f"Predicted Classes (n_classes={n_classes})")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(*scatter.legend_elements(), title="Classes")

    plt.tight_layout()
    plt.show()

def load_iris_data(file_path):
    # Load the Iris dataset
    iris_data = pd.read_csv(file_path)
    X_iris = iris_data.iloc[:, :-1].values  # Features (all but last column)
    y_iris = iris_data.iloc[:, -1].values    # Labels (last column)

    return X_iris, y_iris

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test the RDAModel with synthetic or Iris data.")
    
    # Add arguments for the model parameters
    parser.add_argument('--samples', type=int, default=100, help='Number of samples per class')
    parser.add_argument('--features', type=int, default=4, help='Number of features')
    parser.add_argument('--classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--iris', action='store_true', help='Use Iris dataset instead of synthetic data')

    # Parse arguments
    args = parser.parse_args()

    if args.iris:
        # Load and test the RDAModel on the Iris dataset
        X_iris, y_iris = load_iris_data('iris.csv')  # Ensure 'iris.csv' is in the same directory
        rda_model = RDAModel(alpha=0.5)
        rda_model.fit(X_iris, y_iris)
        y_pred = rda_model.predict(X_iris)
        
        # Visualization for Iris dataset
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("True Classes (Iris Dataset)")
        scatter = plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, cmap='viridis', marker='o', edgecolor='k')
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.legend(*scatter.legend_elements(), title="Classes")

        plt.subplot(1, 2, 2)
        plt.title("Predicted Classes (Iris Dataset)")
        scatter = plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.legend(*scatter.legend_elements(), title="Classes")

        plt.tight_layout()
        plt.show()
    else:
        # Test the synthetic data model
        test_rda_model(n_samples_per_class=args.samples, n_features=args.features, n_classes=args.classes, seed=args.seed)
