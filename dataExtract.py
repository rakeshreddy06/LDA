import pandas as pd

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

