import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')

species = df['Species'].unique()
# In the dictionary iam taking different species data seperately for further calculations
speciesData = {specie: df[df['Species'] == specie] for specie in species}
# this is for storing the corresponding mean values
dataMeanValues = {}


def calculateMean(data):
    valid_cols = [col for col in data.columns if col not in ['Id', 'Species']]
    meanValues = data[valid_cols].mean()
    return meanValues


for specie, data in speciesData.items():
    dataMeanValues[specie] = calculateMean(data)


# calculated overall mean i.e it will consider sataset as whole and give mean values for features which will used in calculating between class scatter
valid_cols = [col for col in df.columns if col not in ['Id', 'Species']]
MeanOverall = df[valid_cols].mean().values  # Mean of all samples

# Method to calculate both S_W and S_B
def calculate_scatter_matrices(speciesData, dataMeanValues, MeanOverall):
    
    #initializations
    No_offeatures = len(valid_cols)  
    WithinScatterMtrx = np.zeros((No_offeatures, No_offeatures)) 
    BetweenScatterMtrx = np.zeros((No_offeatures, No_offeatures))  

    
    for specie, data in speciesData.items():
       
        Extracted_data = data[valid_cols].values

        Vectr_Mean = dataMeanValues[specie].values.reshape(No_offeatures, 1)

        #loop iterates over each sample in the class
        for i in range(Extracted_data.shape[0]):
            x_i = Extracted_data[i].reshape(No_offeatures, 1)  # converting feature array to  matrix
            WithinScatterMtrx  += (x_i - Vectr_Mean).dot((x_i - Vectr_Mean).T)

        # Compute between-class scatter (S_B)
        n_i = Extracted_data.shape[0]  # Number of samples in this class
        Mean_Diference = Vectr_Mean - MeanOverall.reshape(No_offeatures, 1)  # Difference between class mean and overall mean
        BetweenScatterMtrx += n_i *  Mean_Diference.dot( Mean_Diference.T)

    return WithinScatterMtrx, BetweenScatterMtrx

# Call the method to calculate scatter matrices
WithinScatterMtrx, BetweenScatterMtrx = calculate_scatter_matrices(speciesData, dataMeanValues, MeanOverall)

# Print the results
print("Within-class scatter matrix :\n", WithinScatterMtrx)
print("Between-class scatter matrix :\n", BetweenScatterMtrx)




# calculating eigenvalues by formula S_W^-1 * S_B
EigenValues, EigenVectors = np.linalg.eig(np.linalg.inv(WithinScatterMtrx).dot(BetweenScatterMtrx))

# Sort eigenvalues and eigenvectors in descending order
#eigen vectors are linear dicrimininant indocate maximum separation between classes 
Desc_eigen = np.argsort(-EigenValues)  
EigenValues = EigenValues[Desc_eigen]
EigenVectors = EigenVectors[:, Desc_eigen]


# typically top 2 vectors are selected to iris  ....need to make generic code for random data
EV = EigenVectors[:, :2]  #2  eigenvectors with highest discriminant values (for 2D projection)

# project data to 1 dimension
# Assuming X is your original dataset (without 'Species' column)
OriginalX = df[valid_cols].values  # Extract feature data (X)
OriginalX_lda = OriginalX.dot(EV)  # Project the data onto the new discriminant axes

# Print the eigenvalues (discriminative power of each component)
print("Eigenvalues:\n", EigenValues)

# Print the transformed data (in reduced dimensions)
print("Projected data (LDA):\n", OriginalX_lda)



plt.figure(figsize=(8, 6))
for species, marker, color in zip(df['Species'].unique(), ['o', 's', 'D'], ['r', 'g', 'b']):
    plt.scatter(OriginalX_lda[df['Species'] == species, 0], OriginalX_lda[df['Species'] == species, 1],
                marker=marker, color=color, label=species)
plt.title('LDA: Iris projection onto first 2 linear discriminants')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()



