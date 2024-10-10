import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LDAModel:
    def __init__(self, nComponents, reg_param=1e-5):  # Add regularization parameter
        self.nComponents = nComponents
        self.Eigens = None
        self.classes_ = None
        self.classMeans = None
        self.reg_param = reg_param  # Initialize the regularization parameter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        WithinScatterMtrx, BetweenScatterMtrx = self.calculate_scatter_matrices(X, y)

        # Add regularization to the diagonal of WithinScatterMtrx
        WithinScatterMtrx += self.reg_param * np.eye(n_features)

        # Perform the eigen decomposition
        eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(WithinScatterMtrx).dot(BetweenScatterMtrx))
        sorted_indices = np.argsort(-eigenValues.real)
        eigenValues = eigenValues.real[sorted_indices]
        eigenVectors = eigenVectors.real[:, sorted_indices]

        # Select the top components
        if self.nComponents is not None:
            eigenVectors = eigenVectors[:, :self.nComponents]
        else:
            self.nComponents = len(self.classes_) - 1
            eigenVectors = eigenVectors[:, :self.nComponents]

        self.Eigens = eigenVectors

        # Project data onto LDA components
        XLDA = X.dot(self.Eigens)
        self.classMeans = {cls: XLDA[y == cls].mean(axis=0) for cls in self.classes_}

        return self

    # ... rest of the code remains unchanged


    def calculate_scatter_matrices(self, X_train, y_train):
        speciesData = {label: X_train[y_train == label] for label in np.unique(y_train)}
        dataMeanValues = {label: np.mean(speciesData[label], axis=0) for label in speciesData}
        MeanOverall = np.mean(X_train, axis=0).reshape(-1, 1)

        n_features = X_train.shape[1]
        WithinScatterMtrx = np.zeros((n_features, n_features))
        BetweenScatterMtrx = np.zeros((n_features, n_features))

        for label, class_data in speciesData.items():
            n_i = class_data.shape[0]
            class_mean = dataMeanValues[label].reshape(n_features, 1)

            for sample in class_data:
                sample = sample.reshape(n_features, 1)
                WithinScatterMtrx += (sample - class_mean).dot((sample - class_mean).T)

            mean_diff = class_mean - MeanOverall
            BetweenScatterMtrx += n_i * mean_diff.dot(mean_diff.T)

        return WithinScatterMtrx, BetweenScatterMtrx

    def transform(self, X):
        if self.Eigens is None:
            raise ValueError("Model has not been fitted yet.")
        return X.dot(self.Eigens)

    def get_params(self):
        return {
            'nComponents': self.nComponents,
            'classes_': self.classes_,
            'Eigens': self.Eigens,
            'classMeans': self.classMeans
        }


class RDAModelResults:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        XLDA = self.model.transform(X)
        predictions = []
        for sample in XLDA:
            distances = {cls: np.linalg.norm(sample - mean) for cls, mean in self.model.classMeans.items()}
            predicted_class = min(distances, key=lambda k: distances[k])
            predictions.append(predicted_class)
        return np.array(predictions)


def plot_lda_projection(XLDA, y, title):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(XLDA[y == label, 0], XLDA[y == label, 1], label=f'Class {label}')
    plt.title(title)
    plt.xlabel('LD1')
    if XLDA.shape[1] > 1:
        plt.ylabel('LD2')
    plt.legend()
    plt.show()


def generate_random_data(n_samples_per_class=50, n_features=4, n_classes=3, seed=42):
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


def read_csv_data(file_path):
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def standardize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)