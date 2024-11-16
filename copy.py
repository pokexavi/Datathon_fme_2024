import pandas as pd


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
    data['Structure.Cooling'] = data['Structure.Cooling'].apply(lambda x: 0 if pd.isnull(x) or 'none' in str(x) else 1)
    data['Structure.Heating'] = data['Structure.Heating'].apply(lambda x: 0 if pd.isnull(x) or 'none' in str(x) else 1)
    data['cooling_heating'] = data.apply(
    lambda row: 0 if row['Structure.Cooling'] == 0 and row['Structure.Heating'] == 0 else
                (1 if (row['Structure.Cooling'] == 1 and row['Structure.Heating'] == 0) or
                       (row['Structure.Cooling'] == 0 and row['Structure.Heating'] == 1) else 2),
    axis=1
)
    

def treat_new_construction_ny(data):
    """
    If value equals true then 1 else 0
    """
    data['Structure.NewConstructionYN'] = data['Structure.NewConstructionYN'].apply(lambda x: 1 if x == 'true' else 0)


def treat_basement(data):
    """
    If value equals true then 1 else 0
    """
    data['Structure.Basement'] = data['Structure.Basement'].apply(lambda x: 0 if pd.notnull(x) or 'none' in str(x) else (1 if pd.notnull(x) else x))

def treat_fireplace(data):
    """If nan the value is 0"""
    data['Structure.FireplacesTotal'] = data['Structure.FireplacesTotal'].apply(lambda x: 0 if pd.isnull(x) else x)


def treat_garage(data):
    """
    If the value greater than 0 then 1 else 0. If nan then 0
    """
    data['Structure.GarageSpaces'] = data['Structure.GarageSpaces'].apply(lambda x: 0 if pd.isnull(x) else (1 if x > 0 else 0))

def structure_transformations(data):
    """
    Preprocess the features with structure code
    """
    treat_cooling_heating(data)
    treat_new_construction_ny(data)
    treat_basement(data)
    treat_fireplace(data)
    treat_garage(data)

def write_data(data,index):
    """
    Write the data to a csv file
    """
    train = data.iloc[:index]
    test = data.iloc[index:]
    train.to_csv('data/train_preprocessed.csv',index=False)
    test.to_csv('data/test_preprocessed.csv',index=False)

def drop_features(data):
    data.drop(['Structure.Cooling','Structure.Heating','Structure.BelowGradeFinishedArea','Structure.BelowGradeUnfinishedArea'
               'Structure.BathroomsHalf','Structure.BedroomsTotal','Structure.BathroomsFull','Structure.YearBuilt','Structure.ParkingFeatures',],axis=1,inplace=True)



def main():
    train,test = read_data()
    data,index = merge_data(train,test)
    structure_transformations(data)
    location_transformations(data)
    image_transformations(data)
    drop_features(data)

    write_data(data,index)
