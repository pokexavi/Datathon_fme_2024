import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np


LOG_COLUMNS = ['Structure.GarageSpaces', 'Structure.FireplacesTotal','Structure.LivingArea','Structure.Rooms.RoomsTotal']
NUMERICAL = ['ImageData.style.stories.summary.label','Structure.FireplacesTotal','Structure.GarageSpaces','Structure.LivingArea','Structure.Rooms.RoomsTotal',
             'Listing.Price.ClosePrice','Listing.Price.ClosePrice'] 

def log_transformations(data):
    """
    Apply the log transformation to the columns that have a positive value
    """
    for col in LOG_COLUMNS:
        data[col] = data[col].apply(lambda x: np.log1p(x) if x > 0 else x)
    return data

def read_data_processed():
    """
    Read the data from the csv files
    """
    train = pd.read_csv('../data/train_preprocessed.csv')
    test = pd.read_csv('../data/test_preprocessed.csv')
    return train,test

def numeric_outliers(data,numerical):
    """
    Remove the outliers from the data
    """
    numerical = numerical[:-1]
    threshold = 3
    z_scores = np.abs(stats.zscore(data[numerical]))
    mask = np.all(z_scores < threshold, axis=1)
    data = data[mask]
    return data

def remove_outliers(data):
    data = numeric_outliers(data,NUMERICAL)
    return data

def remove_collinear_features(data, threshold=10,max_iterations=10):
    """
    Remove features with high correlation to avoid multicollinearity
    """
    removed_features = []
    iteration = 0

    while True:
        # Calcular el VIF para cada característica
        vif = pd.DataFrame()
        vif["Feature"] = data.columns
        vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        
        # Encontrar la característica con el VIF más alto
        max_vif = vif["VIF"].max()
        if max_vif > threshold and iteration < max_iterations:
            # Identificar la característica a eliminar
            feature_to_remove = vif.loc[vif["VIF"] == max_vif, "Feature"].values[0]
            removed_features.append(feature_to_remove)
            # Eliminar la característica del DataFrame (internamente, no devuelve el DataFrame)
            data = data.drop(columns=[feature_to_remove])
            iteration += 1
        else:
            break

    return removed_features

def write_data_processed(data,phase):
    """
    Write the data to the csv files
    """
    if phase == "train":
        print("Length of train data: ",len(data))
        data.to_csv('../data/train_preprocessed.csv', index=False)
    else:
        data.to_csv('../data/test_preprocessed.csv', index=False)


def transform(data,phase):
    features = []
    if phase == "train":
        data = remove_outliers(data)
    data = log_transformations(data)
    if phase == "train":
        print(data.shape)
        numerical = NUMERICAL[:-1]
        features = remove_collinear_features(data[numerical])
        data.drop(features,axis=1,inplace=True)
        print(data.shape)
    write_data_processed(data,phase)
    return features

def main_transformation():
    train,test = read_data_processed()
    features_1 = transform(train,"train")
    features_2 = transform(test,"test")
    test.drop(features_1,axis=1,inplace=True)
    write_data_processed(test,"test")
    print("Featured removed by colinearity: ",features_1)   

main_transformation()
