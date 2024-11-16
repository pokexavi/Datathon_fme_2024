import pandas as pd


def read_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train,test

def main():
    train,test = read_data()
    index = 



