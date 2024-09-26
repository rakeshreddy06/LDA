import numpy as np
import matplotlib.pyplot as plt


def generate_random_data(n_samples_per_class=50, n_features=4, n_classes=3):
    X = []
    y = []
    
    
    for class_label in range(n_classes):
        class_data = np.random.randn(n_samples_per_class, n_features) + class_label * 2 
        X.append(class_data)
        y.extend([class_label] * n_samples_per_class) 

    
    X = np.vstack(X)
    y = np.array(y)
    
    return X, y


X_synthetic, y_synthetic = generate_random_data(n_samples_per_class=50, n_features=4, n_classes=3)


def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


X_train, X_test, y_train, y_test = manual_train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)


def calculate_scatter_matrices(X_train, y_train):
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


WithinScatterMtrx, BetweenScatterMtrx = calculate_scatter_matrices(X_train, y_train)


EigenValues, EigenVectors = np.linalg.eig(np.linalg.inv(WithinScatterMtrx).dot(BetweenScatterMtrx))


sorted_indices = np.argsort(-EigenValues)
EigenValues = EigenValues[sorted_indices]
EigenVectors = EigenVectors[:, sorted_indices]


EV = EigenVectors[:, :2]


X_train_lda = X_train.dot(EV)
X_test_lda = X_test.dot(EV)


print("Eigenvalues:\n", EigenValues)


print("Projected training data (LDA):\n", X_train_lda)


print("Projected testing data (LDA):\n", X_test_lda)


plt.figure(figsize=(8, 6))
for label in np.unique(y_train):
    plt.scatter(X_train_lda[y_train == label, 0], X_train_lda[y_train == label, 1], label=f'Class {label}')
plt.title('LDA Projection of Training Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
for label in np.unique(y_test):
    plt.scatter(X_test_lda[y_test == label, 0], X_test_lda[y_test == label, 1], label=f'Class {label}')
plt.title('LDA Projection of Testing Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()



#task done: 
# 1) calculated eigen values and vectors.
# 2) converted 2 dimensional to 1 D 
# 3) removed iris dataset and added  randondom data generator for both training testing.
# 4) added visualization using plots for both training and testing.

#todo:
#1) rename functions and variables in random data generator code.
#2) add comments whereever you feel like
#3) try to modify data generation logic in random data generation function like see whats noise parameter etc..
#4) check if code works for k classes if not modify it for k classes

