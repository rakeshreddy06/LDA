import argparse
import numpy as np
import pandas as pd
from lda_model import LDAModel, RDAModelResults, plot_lda_projection
import matplotlib.pyplot as plt


# Function to generate synthetic data
def generate_synthetic_data(n_samples_per_class=50, n_features=4, n_classes=3, seed=42):
    np.random.seed(seed)
    X = []
    y = []
    for class_label in range(n_classes):
        class_data = np.random.randn(n_samples_per_class, n_features) + class_label * 2
        X.append(class_data)
        y.extend([class_label] * n_samples_per_class)

    X = np.vstack(X)
    y = np.array(y)

    return X, y


def plot_true_vs_predicted(XLDA, true_labels, predicted_labels):
    """
    Plot true and predicted class projections side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for True Labels
    axes[0].set_title("True Class Projections")
    for label in np.unique(true_labels):
        axes[0].scatter(XLDA[true_labels == label, 0], XLDA[true_labels == label, 1], label=f'Class {label}')
    axes[0].set_xlabel('LD1')
    axes[0].set_ylabel('LD2')
    axes[0].legend()

    # Plot for Predicted Labels
    axes[1].set_title("Predicted Class Projections")
    for label in np.unique(predicted_labels):
        axes[1].scatter(XLDA[predicted_labels == label, 0], XLDA[predicted_labels == label, 1], label=f'Class {label}')
    axes[1].set_xlabel('LD1')
    axes[1].set_ylabel('LD2')
    axes[1].legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['synthetic', 'csv'], required=True, help='Type of data to test the model on')
    parser.add_argument('--file', type=str, help='Path to the CSV file (required if data type is csv)')
    parser.add_argument('--n_components', type=int, default=2, help='Number of components to project onto')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples per class for synthetic data')
    parser.add_argument('--n_features', type=int, default=4, help='Number of features for synthetic data')
    parser.add_argument('--n_classes', type=int, default=3, help='Number of classes for synthetic data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for synthetic data generation')
    args = parser.parse_args()

    if args.data == 'synthetic':
        X, y = generate_synthetic_data(n_samples_per_class=args.n_samples, n_features=args.n_features,
                                       n_classes=args.n_classes, seed=args.seed)
    elif args.data == 'csv':
        if not args.file:
            raise ValueError("CSV file path must be provided for 'csv' data type.")
        df = pd.read_csv(args.file, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    # Create LDA model
    model = LDAModel(nComponents=args.n_components)
    model.fit(X, y)

    # Transform the data using LDA
    X_lda = model.transform(X)

    # Perform predictions
    results = RDAModelResults(model)
    predictions = results.predict(X)

    # Plot true and predicted class projections
    plot_true_vs_predicted(X_lda, y, predictions)


if __name__ == "__main__":
    main()
