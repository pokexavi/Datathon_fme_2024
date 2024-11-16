import pandas as pd
from add_zhvi import add_zhvi_values


def read_data():
    """
    Read the data from the csv files
    """
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train,test

def merge_data(train,test):
    """
    Merge the train and test data and return the index of the first test observation in the merged data
    """
    data = pd.concat([train,test],axis=0)
    index = len(train)
    return data, index

def treat_cooling_heating(data):
    """
    We consider taht a house does not have cooling or heating if the value is none
    """
    data['Structure.Cooling'] = data['Structure.Cooling'].apply(lambda x: 0 if pd.notnull(x) and 'none' in str(x) else (1 if pd.notnull(x) else x))
    data['Structure.Heating'] = data['Structure.Heating'].apply(lambda x: 0 if pd.notnull(x) and 'none' in str(x) else (1 if pd.notnull(x) else x))
    #New column
    #si no tiene nada 0, si tiene una de las dos un 1 sino 0

    data['cooling_heating'] = 


def structure_transformations(data):
    """
    Preprocess the features with structure code
    """
    treat_cooling_heating(data)
    treat_new_construction_ny(data)

def write_data(data,index):
    """
    Write the data to a csv file
    """
    train = data.iloc[:index]
    test = data.iloc[index:]
    train.to_csv('data/train.csv',index=False)
    test.to_csv('data/test.csv',index=False)

def seasonality_transformations(data):
    """
    Add seasonality features to the data
    """
    data['month'] = data['Listing.Dates.CloseDate'].dt.month
    data.drop(['Tax.Zoning','UnitTypes,UnitTypeType'],axis=1)



def main():
    train,test = read_data()
    data,index = merge_data(train,test)
    add_zhvi_values(data)
    structure_transformations(data)
    location_transformations()
    image_transformations()
    seasonality_transformations(data)

    write_data(data,index)



