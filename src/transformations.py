import pandas as pd
from scipy import stats
import numpy as np


LOG_COLUMNS = ['Structure.GarageSpaces', 'Structure.FireplacesTotal']
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
    train = pd.read_csv('data/train_preprocessed.csv')
    test = pd.read_csv('data/test_preprocessed.csv')
    return train,test

def numeric_outliers(data):
    """
    Remove the outliers from the data
    """
    threshold = 3
    for column in data.select_dtypes(include=['number']).columns:
        z_scores = np.abs(stats.zscore(data[column]))
        data = data[(z_scores < threshold)]
    return data

def categorical_outliers(data):
    """
    Remove the outliers from the data
    """
    threshold_freq = 0.01 * len(data)
    for column in data.select_dtypes(include=['object']).columns:
        data = data[data.groupby(column)[column].transform('count') > threshold_freq]
    return data

def remove_outliers(data):
    data = numeric_outliers(data)
    data = categorical_outliers(data)
    return data

def remove_collinear_features(data):
    """
    Remove features with high correlation to avoid multicollinearity
    """

def write_data_processed(train, test):
    """
    Write the data to the csv files
    """
    train.to_csv('data/train_preprocessed.csv', index=False)
    test.to_csv('data/test_preprocessed.csv', index=False)


def transform(data,phase):
    features = []
    if phase == "train":
        data = remove_outliers(data)
    data = log_transformations(data)
    if phase == "train":
        features = remove_collinear_features(data)
    write_data_processed(data)
    return features

def main_transformation():
    train,test = read_data_processed()
    features = transform(train,"train")
    test.drop(features,axis=1,inplace=True)
    features = transform(test,"test")