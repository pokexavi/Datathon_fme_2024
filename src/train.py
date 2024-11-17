import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import multiprocessing as mp
import numpy as np
import mlflow
import datetime
import os



def read_data_processed():
    """
    Read the data from the csv files
    """
    train = pd.read_csv('../data/train_preprocessed.csv')
    test = pd.read_csv('../data/test_preprocessed.csv')
    return train,test

def create_param_grid():
    """
    Create the parameter grid for the RandomizedSearchCV
    """
    return {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],  
        'max_depth': [None] + [int(x) for x in np.linspace(10, 100, num=10)],         
        'min_samples_split': [2, 5, 10],                                              
        'min_samples_leaf': [1, 2, 4],                                               
        'max_features': ['auto', 'sqrt', 'log2'],                                    
        'bootstrap': [True, False]                                                   
    }



def create_experiment(name='housing_price_prediction'):
    """
    Create the experiment in MLflow
    """
    try:
        mlflow.set_experiment(name)
    except mlflow.exceptions.MlflowException as e:
        mlflow.create_experiment(name)
        mlflow.set_experiment(name)
    

def prepare_data_to_train(data):
    """
    Prepare the data to train the model
    """
    X = data.drop(columns=['Listing.Price.ClosePrice'])
    y = data['Listing.Price.ClosePrice']

    return X, y

def save_feature_importance(model, feature_names, run_id):
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    #si no existe la carpeta data, se crea
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    feature_imp.to_csv('temp_data/feature_importances'+run_id+'.csv', index=False)
    mlflow.log_artifact('temp_data/feature_importances'+run_id+'.csv')
    return feature_imp


def train_random_forest(x_train,y_train):
    """
    Train a Random Forest model using RandomizedSearchCV
    """
    base_model = RandomForestRegressor(
        oob_score=True,
        random_state=42,
        n_jobs=mp.cpu_count()
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=create_param_grid(), 
        n_iter=10,                               
        cv=5,                                    
        n_jobs=mp.cpu_count(),                    
        verbose=2,                                
        scoring='neg_mean_squared_error',
    )         
    random_search.fit(x_train, y_train)
    best_model = random_search.best_estimator_

    id = "Model_" + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with mlflow.start_run(run_name=id) as run:
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metrics(random_search.best_score_)
        mlflow.sklearn.log_model(best_model, id)
        #añade tag que es random forest
        mlflow.set_tag("model", "RandomForest") 
        feature_names = x_train.columns 
        feature_imp = save_feature_importance(best_model, feature_names, id)

def prediction(x_test,y_test):
    """
    Predict the values using the best model
    """
    best_model = mlflow.search_runs(filter_string="tags.model = 'RandomForest'").sort_values(by=['metrics.mean_squared_error'], ascending=True).iloc[0]
    predictions = best_model.predict(x_test)
    return predictions


def main_train():
    TRACKING_MLFLOW = "http://localhost:5000"
    mlflow.set_tracking_uri(TRACKING_MLFLOW)
    create_experiment()
    train,x_test= read_data_processed()
    x_train,y_train = prepare_data_to_train(train)
    train_random_forest(x_train,y_train)
    predictions = prediction(x_test)
    #guarda en la carpeta predictions y mira si existe
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    pd.DataFrame(predictions).to_csv('predictions/predictions.csv', index=False)
