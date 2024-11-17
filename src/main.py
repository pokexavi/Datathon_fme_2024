from preprocessing import main_data_preprocessing
from transformations import main_transformations
from train import main_train

if "__main__" == __name__:
    main_data_preprocessing()
    main_transformations()
    main_train()