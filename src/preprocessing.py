import pandas as pd
import re
from add_zhvi import add_zhvi_values


def read_data():
    """
    Read the data from the csv files
    """
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    return train, test


def merge_data(train, test):
    """
    Merge the train and test data and return the index of the first test observation in the merged data
    """
    data = pd.concat([train, test], axis=0)
    index = len(train)
    return data, index


def treat_cooling_heating(data):
    """
    We consider that a house does not have cooling or heating if the value is none
    """
    data['Structure.Cooling'] = data['Structure.Cooling'].apply(
        lambda x: 0 if pd.isnull(x) or 'none' in str(x) else 1)
    data['Structure.Heating'] = data['Structure.Heating'].apply(
        lambda x: 0 if pd.isnull(x) or 'none' in str(x) else 1)
    data['cooling_heating'] = data.apply(
        lambda row: 0 if row['Structure.Cooling'] == 0 and row['Structure.Heating'] == 0 else
        (1 if (row['Structure.Cooling'] == 1 and row['Structure.Heating'] == 0) or
         (row['Structure.Cooling'] == 0 and row['Structure.Heating'] == 1) else 2),
        axis=1)


def treat_new_construction_ny(data):
    """
    If value equals true then 1 else 0
    """
    data['Structure.NewConstructionYN'] = data['Structure.NewConstructionYN'].apply(lambda x: 1 if x == 'true' else 0)


def treat_basement(data):
    """
    If value equals true then 1 else 0
    """
    data['Structure.Basement'] = data['Structure.Basement'].apply(
        lambda x: 0 if pd.isnull(x) or 'none' in str(x) else (1 if pd.notnull(x) else x))


def treat_fireplace(data):
    """If nan the value is 0"""
    data['Structure.FireplacesTotal'] = data['Structure.FireplacesTotal'].fillna(0)


def treat_garage(data):
    """
    If the value is nan then 0, if is greater than 0 minimum 1 if not round the value
    """
    data['Structure.GarageSpaces'] = data['Structure.GarageSpaces'].apply(
        lambda x: 0 if pd.isnull(x) else (1 if x > 0 and x < 1 else round(x)))


def structure_transformations(data):
    """
    Preprocess the features with structure code
    """
    treat_cooling_heating(data)
    treat_new_construction_ny(data)
    treat_basement(data)
    treat_fireplace(data)
    treat_garage(data)


def characteristics_transformations(data):
    """
    Preprocess the features with characteristics code
    """
    pattern = r"'([^']+)'"
    unique_words = set()

    for entry in data['Characteristics.LotFeatures']:
        if isinstance(entry, str) and entry != 'nan':  # Skip 'nan' and non-string entries
            matches = re.findall(pattern, entry)
            unique_words.update(matches)

    substring_count = {substring: 0 for substring in unique_words}

    for substring in unique_words:
        substring_count[substring] = data['Characteristics.LotFeatures'].str.contains(substring).sum()

    categories = [key for key, value in substring_count.items() if value > 1000]

    for category in categories:
        column_name = f"{category}_Flag"
        data[column_name] = data['Characteristics.LotFeatures'].str.contains(category).astype(float)
        data[column_name].fillna(0.0, inplace=True)


def location_transformations(data):
    """
    Preprocess the features with location code
    """
    data.drop(columns=[
        "Location.Area.SubdivisionName", "Location.School.HighSchoolDistrict",
        "Location.Address.StreetNumber", "Location.Address.CountyOrParish",
        "Location.Address.CensusTract", "Location.Address.City",
        "Location.Address.CensusBlock", "Location.Address.PostalCodePlus4",
        "Location.Address.StateOrProvince", "Location.Address.StreetDirectionPrefix",
        "Location.Address.StreetDirectionSuffix", "Location.Address.StreetName",
        "Location.Address.StreetSuffix", "Location.Address.UnitNumber",
        "Location.Address.UnparsedAddress", 'Location.GIS.Latitude', 'Location.GIS.Longitude',
        'Tax.Zoning', 'UnitTypes.UnitTypeType'], inplace=True)


def image_transformations(data):
    """
    Transform the image data
    """
    data["ImageData.c1c6.Mean"] = data[[
        'ImageData.c1c6.summary.bathroom', 'ImageData.c1c6.summary.exterior',
        'ImageData.c1c6.summary.interior', 'ImageData.c1c6.summary.kitchen',
        'ImageData.c1c6.summary.property']].mean(axis=1)
    data["ImageData.q1q6.Mean"] = data[[
        'ImageData.q1q6.summary.bathroom', 'ImageData.q1q6.summary.exterior',
        'ImageData.q1q6.summary.interior', 'ImageData.q1q6.summary.kitchen',
        'ImageData.q1q6.summary.property']].mean(axis=1)

    data.drop(columns=[
        'ImageData.c1c6.summary.bathroom', 'ImageData.c1c6.summary.exterior',
        'ImageData.c1c6.summary.interior', 'ImageData.c1c6.summary.kitchen',
        'ImageData.c1c6.summary.property', 'ImageData.q1q6.summary.bathroom',
        'ImageData.q1q6.summary.exterior', 'ImageData.q1q6.summary.interior',
        'ImageData.q1q6.summary.kitchen', 'ImageData.q1q6.summary.property'], inplace=True)

    data["ImageData.style.stories.summary.label"] = data["ImageData.style.stories.summary.label"].str.split('_').str[0].astype(float)

    pattern = r"'([^']+)'"
    unique_words = set()

    for entry in data['ImageData.features_reso.results']:
        if isinstance(entry, str) and entry != 'nan':
            matches = re.findall(pattern, entry)
            unique_words.update(matches)

    substring_count = {substring: 0 for substring in unique_words}

    for substring in unique_words:
        substring_count[substring] = data['ImageData.features_reso.results'].str.contains(substring).sum()

    categories = [key for key, value in substring_count.items() if value > 1000]
    important_features = ["View", "CommunityFeatures", "AssociationAmenities",
                          "WaterfrontFeatures", "PoolFeatures", "SpaFeatures"]
    important_categories = [item for item in categories if any(sub in item for sub in important_features)]

    for category in important_categories:
        column_name = f"{category}_Flag"
        data[column_name] = data['ImageData.features_reso.results'].str.contains(category).astype(float)
        data[column_name].fillna(0.0, inplace=True)


def seasonality_transformations(data):
    """
    Add seasonality features to the data
    """
    data['Listing.Dates.CloseDate'] = pd.to_datetime(data['Listing.Dates.CloseDate'], errors='coerce')
    data['Month'] = data['Listing.Dates.CloseDate'].dt.month
    data.drop(columns='Listing.Dates.CloseDate', inplace=True)


def drop_features(data):
    """
    Drop unused features
    """
    data.drop(['Structure.Cooling', 'Structure.Heating', 'Structure.BelowGradeFinishedArea',
               'Structure.BelowGradeUnfinishedArea', 'Structure.BathroomsHalf',
               'Structure.BedroomsTotal', 'Structure.BathroomsFull', 'Structure.YearBuilt',
               'Structure.ParkingFeatures', 'Characteristics.LotFeatures',
               'Characteristics.LotSizeSquareFeet', 'ImageData.room_type_reso.results',
               'ImageData.features_reso.results'], axis=1, inplace=True)


def transformation_features(data):
    add_zhvi_values(data)
    structure_transformations(data)
    characteristics_transformations(data)
    location_transformations(data)
    image_transformations(data)
    seasonality_transformations(data)
    drop_features(data)


def write_data(data, index):
    """
    Write the data to a csv file
    """
    train = data.iloc[:index]
    test = data.iloc[index:]
    train.to_csv('../data/train_preprocessed.csv', index=False)
    test.to_csv('../data/test_preprocessed.csv', index=False)


def main_preprocessing():
    train, test = read_data()
    data, index = merge_data(train, test)
    transformation_features(data)
    write_data(data, index)

