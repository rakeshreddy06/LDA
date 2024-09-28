# LDA
Project work on LDA classification

Why to use LDA?
We LDA to reduce detention(features). In the real world there will be many features. To reduce the number of features  and preserve the discriminatory information.

In LDA we need to follow step by step process to find the object class
Step 1: Extract the data from Iris
Step 2: Calculate the mean for each class and each feature seperately
Step 3: find the inClass scatter matrix
Step 4: find the interclass scatter matrix
step 5:


InClass scatter matrix
we need to use  Sw = Σ(i=1 to c) Σ(x in class i) (x - μi)(x - μi)^T for finding Inclass Scatter matrix.

Sw meant scatter matrix
c is the number of classes
x is the sample in class i
μi is the mean of the class


BetweenClass scatter matrix

we need to use SB = Σ(Ni * (μi - μ) * (μi - μ)^T)  for finding Between Class Scatter matrix

Ni is the number of samples in a class
μi mean of class i









