import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Iris.csv')

species = df['Species'].unique()
# In the dictionary iam taking different species data seperately for further calculations
speciesData = {specie: df[df['Species'] == specie] for specie in species}
# this is for storing the corresponding mean values
dataMeanValues = {}


def calculateMean(data):
    validCols = [col for col in data.columns if col not in ['Id', 'Species']]
    meanValues = data[validCols].mean()
    return meanValues


for specie, data in speciesData.items():
    dataMeanValues[specie] = calculateMean(data)


def calculateClassScatter(data,mean):
    validCols = [col for col in data.columns if col not in ['Id', 'Species']]
    validColsValues=data[validCols].values
    mean= mean.values.reshape(1,-1)
    difference= validColsValues-mean
    return np.dot(difference.T,difference)





# we have 4 features so that i initialize it 4,4
inClassScatter= np.zeros((4,4))
for specie, data in speciesData.items():
    # calculate the inClass for each specie class
    classScatter= calculateClassScatter(data,dataMeanValues[specie])
    inClassScatter+=classScatter

print(inClassScatter)
# we are claculating overallMean of the iris dataset
overallMean= calculateMean(df)

betweenClassScatter= np.zeros((4,4))

for specie,data in speciesData.items():
    N= len(data)
    meanDifference=(dataMeanValues[specie]-overallMean).values.reshape(-1,1)
    betweenClassScatter+=N*np.dot(meanDifference,meanDifference.T)

print(betweenClassScatter)

# compute the eigenvectors and eigenvalues of the matrix
eigenValues, eigenVectors = np.linalg.eig(np.dot(np.linalg.inv(inClassScatter), betweenClassScatter))




# I Created a list of (eigenvalue, eigenvector) tuples
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:, i]) for i in range(len(eigenValues))]

# I want to sort the (eigenvalue, eigenvector) tuples from high to low
eigenPairs = sorted(eigenPairs, key=lambda k: k[0], reverse=True)


print("Sorted Eigenvalues in descending order:")
for pair in eigenPairs:
  print(pair[0])

# I Choosed the top k eigenvectors (k=2 for 2D plot)
k = 2
W = np.hstack([eigenPairs[i][1].reshape(4, 1) for i in range(k)])

print("Projection Matrix (W):\n", W)


features = [col for col in df.columns if col not in ['Id', 'Species']]
X = df[features].values

X_lda = X.dot(W)


lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
lda_df['Species'] = df['Species']

# plotting the data
sns.set(style='whitegrid', palette='muted')


plt.figure(figsize=(10, 7))
sns.scatterplot(
  x='LD1',
  y='LD2',
  hue='Species',
  data=lda_df,
  palette=['r', 'g', 'b'],
  s=100,
  alpha=0.7
)

plt.title('LDA: Iris projection onto the first 2 linear discriminants')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend(title='Species')
plt.show()








    






