import pandas as pd
import numpy as np

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

