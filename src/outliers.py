import numpy as np
from scipy import stats
import pandas as pd


def main_numeric_outliers(data):
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        if data[column].std() > 0:  # Solo procesar si la desviación estándar es mayor a 0
            data = data[(np.abs(stats.zscore(data[column])) < 3)]
    return data


def main_categorical_outliers(data):
    # Descarte de outliers de el dataframe
    # por cada columna categorica del dataframe en funcion a la frecuencia
    # de cada categoria
    threshold_freq = 0.01 * len(data)
    for column in data.select_dtypes(include=['object']).columns:
        data = data[data.groupby(column)[column].transform('count') > threshold_freq]
    return data


def remove_outliers(data):
    data = main_numeric_outliers(data)
    data = main_categorical_outliers(data)
    return data
