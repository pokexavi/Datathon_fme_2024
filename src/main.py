from preprocessing import main_data_preprocessing
from transformations import main_transformations
from train import main_train

if "__main__" == __name__:
    PREPROCESSING = False
    TRANSFORMATIONS = False
    TRAIN = True
    if PREPROCESSING:
        main_data_preprocessing()
    if TRANSFORMATIONS:
        main_transformations()
    if TRAIN:
        main_train()