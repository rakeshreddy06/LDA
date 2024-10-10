# README

## Overview

This project implements a Linear Discriminant Analysis (LDA) model for dimensionality reduction and classification. The model projects high-dimensional data into a lower-dimensional space, maximizing the distance between classes. It is particularly useful for visualizing data with a reduced number of features.

### When to Use the Model

- **Classification Tasks**: Suitable for classifying data into different classes.
- **Multivariate Normal Distribution Assumption**: Effective when data for each class follows a normal distribution.
- **Dimensionality Reduction**: Reduces data to lower dimensions while maintaining class separability.

## Running the Program

### Step 1: Generate the Data

To generate synthetic data, use the following command:
bash
py .\testDataGenerator.py -N <Sample size> -f <features> -c <classes> -seed 10000 -output_file generated_data.csv


Example:

bash
py .\testDataGenerator.py -N 150 -f 8 -c 5 -seed 10000 -output_file generated_data.csv


### Step 2: Run the Program

Execute the final LDA model with:

bash
python final


## Model Testing

### Step 1: Training on Available Dataset (Iris)

- Initially, the model was built using the Iris dataset, a common classification problem involving classifying flowers into one of three different iris species.

### Step 2: Generated Data

- The model was tested using synthetic data generated from `data_generator.py`, allowing for testing under various conditions with variable classes, samples, and features.

### Step 3: Manual Train-Test Split

- Data is split into training and test sets to evaluate performance, preventing overfitting and ensuring accurate training.

### Step 4: Accuracy Calculation

- The model’s predicted classes are compared against the true classes to calculate accuracy.

## User-Exposed Parameters

Users can adjust the following parameters to tune model performance:

- **nComponents**: Number of components to keep in LDA, controlling dimensionality reduction.
- **Regularization**: Adds regularization to the within-scatter matrix, preventing overfitting in high-dimensional data.
- **Solver**: Allows choice of eigenvalue computation method, introducing flexibility.
- **Test Size**: Determines the percentage of data allocated for testing.
- **Random State**: Ensures reproducibility of train-test splits and generated data.

## Known Limitations

The implementation may encounter difficulties with:

1. **Highly Imbalanced Data**: Struggles when class sizes are noticeably imbalanced.
2. **High Dimensionality with Few Samples**: May exhibit irregular behavior when the number of features exceeds the number of samples. Dimensionality reduction using PCA can be a solution.
