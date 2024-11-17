import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import multiprocessing as mp
import numpy as np
import mlflow
import xgboost as xgb
import datetime
import os

def read_data_processed():
    train = pd.read_csv('../data/train_preprocessed.csv')
    test = pd.read_csv('../data/test_preprocessed.csv')
    # Eliminar las columnas innecesarias
    train = train.drop(columns=['Listing.ListingId','Location.Address.PostalCode'])
    test = test.drop(columns=['Location.Address.PostalCode','Listing.Price.ClosePrice'])
    return train, test

def create_param_grid_random_forest():
    return {
        'n_estimators': [int(x) for x in np.linspace(20, 200, 10)],  
        'max_depth': [int(x) for x in np.linspace(3, 40, 10)],         
        'min_samples_split': [5, 10],                                              
        'min_samples_leaf': [2, 4],                                               
        'max_features': ['sqrt', 'log2']                                                                                      
    }

def create_param_grid_xgboost():
    return {
        'n_estimators': [int(x) for x in np.linspace(50, 300, 6)],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

def create_param_grid_catboost():
    return {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'bagging_temperature': [0.1, 0.5, 1],
        'border_count': [32, 64, 128]
    }

def create_experiment(name='housing_price_prediction'):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name)
    mlflow.set_experiment(name)

def prepare_data_to_train(data):
    X = data.drop(columns=['Listing.Price.ClosePrice'])
    y = data['Listing.Price.ClosePrice']
    return X, y

def save_feature_importance(model, feature_names, run_id):
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    file_path = f'temp_data/feature_importances_{run_id}.csv'
    feature_imp.to_csv(file_path, index=False)
    mlflow.log_artifact(file_path)
    return feature_imp

def train_random_forest(x_train, y_train):
    """
    Train a random forest model
    """
    base_model = RandomForestRegressor(
        bootstrap=True,
        random_state=42,
        n_jobs=mp.cpu_count()
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=create_param_grid_random_forest(), 
        n_iter=5,
        cv=5,                                  
        n_jobs=mp.cpu_count(),                    
        verbose=2,                                
        scoring='neg_mean_absolute_error' 
    )         
    
    print("Training random forest...")
    random_search.fit(x_train, y_train)
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(x_train)
    r2 = r2_score(y_train, y_pred)

    # Registrar en MLflow
    run_id = f"Model_{len(mlflow.search_runs())}"
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("mae_score", -random_search.best_score_)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.set_tag("model", "RandomForest")
        mlflow.set_tag("name", run_id)
        feature_names = x_train.columns
        save_feature_importance(best_model, feature_names, run_id)

def train_xgboost(x_train, y_train):
    """
    Train a XGBoost model
    """
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=mp.cpu_count()
    )
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=create_param_grid_xgboost(), 
        n_iter=10,
        cv=5,                                  
        n_jobs=mp.cpu_count(),                    
        verbose=2,                                
        scoring='neg_mean_absolute_error' 
    )
    print("Training XGBoost...")
    random_search.fit(x_train, y_train)
    best_model = random_search.best_estimator_
    run_id = f"Model_{len(mlflow.search_runs())}"

    y_pred = best_model.predict(x_train)
    r2 = r2_score(y_train, y_pred)

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("mae_score", -random_search.best_score_)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("name", run_id)
        feature_names = x_train.columns
        save_feature_importance(best_model, feature_names, run_id)

def train_catboost(x_train, y_train):
    """
    Train a CatBoost model
    """
    base_model = CatBoostRegressor(
        random_state=42,
        thread_count=mp.cpu_count()
    )
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=create_param_grid_catboost(), 
        n_iter=10,
        cv=5,                                  
        n_jobs=mp.cpu_count(),                    
        verbose=2,                                
        scoring='neg_mean_absolute_error' 
    )
    print("Training CatBoost...")
    random_search.fit(x_train, y_train)
    best_model = random_search.best_estimator_
    run_id = f"Model_{len(mlflow.search_runs())}"

    y_pred = best_model.predict(x_train)
    r2 = r2_score(y_train, y_pred)

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("mae_score", -random_search.best_score_)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.set_tag("model", "CatBoost")
        mlflow.set_tag("name", run_id)
        feature_names = x_train.columns
        save_feature_importance(best_model, feature_names, run_id)

def train_ensemble(x_train, y_train):
    """
    Train an ensemble model
    """



def prediction(x_test):
    print("Predicting the values...")
    best_run = mlflow.search_runs(order_by=["metrics.mae_score ASC"], max_results=1)
    best_run_id = best_run.iloc[0].run_id
    experiment_id = best_run.iloc[0].experiment_id
    print(f"Best run id: {best_run_id}, Experiment id: {experiment_id}")
    #id = mlflow.get_run(best_run_id).data.tags['name']

    model_uri = f"mlflow-artifacts:/{experiment_id}/{best_run_id}/artifacts/model"
    best_model = mlflow.sklearn.load_model(model_uri)
    predictions = best_model.predict(x_test)
    return predictions

def main_train():
    TRACKING_MLFLOW = "http://localhost:5000"
    mlflow.set_tracking_uri(TRACKING_MLFLOW)
    
    print("Creating experiment...")
    create_experiment()
    
    train, test = read_data_processed()
    print("Preparing data to train...")
    x_train, y_train = prepare_data_to_train(train)
    
    print("Training random forest...")
    train_random_forest(x_train, y_train)
    train_xgboost(x_train, y_train)
    train_catboost(x_train, y_train)
    #train_ensemble(x_train, y_train)

    
    x_test_to_pred = test.drop(columns=['Listing.ListingId'])
    predictions = prediction(x_test_to_pred)
    
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    predictions_df = pd.DataFrame({
    'Listing.ListingId': test['Listing.ListingId'],  # Añadir la columna 'Listing.ListingId' de 'test'
    'PredictedPrice': predictions                     # Añadir las predicciones
})
    predictions_df.to_csv('predictions/predictions.csv', index=False)
    print("Predictions saved successfully.")

main_train()
