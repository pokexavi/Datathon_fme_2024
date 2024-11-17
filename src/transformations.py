import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np


LOG_COLUMNS = ['Structure.GarageSpaces', 'Structure.FireplacesTotal']
NUMERICAL =['ImageData.style.stories.summary.label','Structure.FireplacesTotal','Structure.GarageSpaces',
             'Structure.LivingArea','Structure.Rooms.RoomsTotal'] 

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
    threshold = 3
    z_scores = np.abs(stats.zscore(data[numerical]))
    mask = np.all(z_scores < threshold, axis=1)
    data = data[mask]
    return data

def remove_outliers(data):
    data = numeric_outliers(data,NUMERICAL)
    return data

def remove_collinear_features(data, threshold=10):
    """
    Remove features with high correlation to avoid multicollinearity
    """
    removed_features = []

    # Repetir hasta que no queden características con VIF mayor que el umbral
    while True:
        # Calcular el VIF para cada característica
        vif = pd.DataFrame()
        vif["Feature"] = data.columns
        vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        
        # Encontrar la característica con el VIF más alto
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            # Eliminar la característica con el VIF más alto
            feature_to_remove = vif.loc[vif["VIF"] == max_vif, "Feature"].values[0]
            data = data.drop(columns=[feature_to_remove])
            removed_features.append(feature_to_remove)
        else:
            break
    
    return data, removed_features


def write_data_processed(train, test):
    """
    Write the data to the csv files
    """
    train.to_csv('../data/train_preprocessed.csv', index=False)
    test.to_csv('../data/test_preprocessed.csv', index=False)


def transform(data,phase):
    features = []
    if phase == "train":
        data = remove_outliers(data)
    data = log_transformations(data)
    if phase == "train":
        data, features = remove_collinear_features(data[NUMERICAL])
    write_data_processed(data)
    return features

def main_transformation():
    train,test = read_data_processed()
    features = transform(train,"train")
    test.drop(features,axis=1,inplace=True)
    features = transform(test,"test")
    print("Featured removed by colinearity: ",features)    