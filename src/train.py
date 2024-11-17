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
        n_iter=30,
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
        mlflow.log_metric("mae_error", -random_search.best_score_)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.set_tag("model", "RandomForest")
        mlflow.set_tag("name", run_id)
        feature_names = x_train.columns
        save_feature_importance(best_model, feature_names, run_id)
    return best_model,run_id

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
        n_iter=50,
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
        mlflow.log_metric("mae_error", -random_search.best_score_)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("name", run_id)
        feature_names = x_train.columns
        save_feature_importance(best_model, feature_names, run_id)
    return best_model,run_id

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
        n_iter=50,
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
        mlflow.log_metric("mae_error", -random_search.best_score_)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.set_tag("model", "CatBoost")
        mlflow.set_tag("name", run_id)
        feature_names = x_train.columns
        save_feature_importance(best_model, feature_names, run_id)
    return best_model,run_id


def train_ensemble(x_train, y_train, model_1, model_2, model_3,id_1,id_2,id_3):
    """
    Train an ensemble model by averaging predictions from RandomForest, XGBoost, and CatBoost
    """
    print("Training ensemble model...")

    # Realizar predicciones con cada modelo en el conjunto de entrenamiento
    predictions_1 = model_1.predict(x_train)
    predictions_2 = model_2.predict(x_train)
    predictions_3 = model_3.predict(x_train)

    # Promediar las predicciones de los tres modelos
    ensemble_predictions = (predictions_1 + predictions_2 + predictions_3) / 3

    # Calcular las métricas de rendimiento
    r2 = r2_score(y_train, ensemble_predictions)
    mae = np.mean(np.abs(y_train - ensemble_predictions))

    # Registrar el ensemble en MLflow
    run_id = f"EnsembleModel_{len(mlflow.search_runs())}"
    with mlflow.start_run(run_name=run_id):
        mlflow.log_metric("mae_error", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.set_tag("model", "EnsembleModel")
        mlflow.set_tag("name", run_id)
        mlflow.set_tag("model_1", id_1)
        mlflow.set_tag("model_2", id_2)
        mlflow.set_tag("model_3", id_3)
    
    print("Ensemble model training completed.")


def train_best_ensemble(x_train, y_train):
    # Obtener los mejores modelos de cada tipo en funcion del tag
    best_rf = mlflow.search_runs(filter_string="tags.model='RandomForest'", order_by=["metrics.mae_error ASC"], max_results=1)
    best_xgb = mlflow.search_runs(filter_string="tags.model='xgboost'", order_by=["metrics.mae_error ASC"], max_results=1)
    best_cat = mlflow.search_runs(filter_string="tags.model='CatBoost'", order_by=["metrics.mae_error ASC"], max_results=1)

    experiment_id = best_rf.iloc[0].experiment_id

    best_rf_id = best_rf.iloc[0].run_id
    model_uri_rf = f"mlflow-artifacts:/{experiment_id}/{best_rf_id}/artifacts/model"
    best_rf = mlflow.sklearn.load_model(model_uri_rf)   

    best_xgb_id = best_xgb.iloc[0].run_id
    model_uri_xgb = f"mlflow-artifacts:/{experiment_id}/{best_xgb_id}/artifacts/model"
    best_xgb = mlflow.sklearn.load_model(model_uri_xgb)

    best_cat_id = best_cat.iloc[0].run_id
    model_uri_cat = f"mlflow-artifacts:/{experiment_id}/{best_cat_id}/artifacts/model"
    best_cat = mlflow.sklearn.load_model(model_uri_cat)

    predictions_1 = best_rf.predict(x_train)
    predictions_2 = best_xgb.predict(x_train)
    predictions_3 = best_cat.predict(x_train)

    # Promediar las predicciones de los tres modelos
    ensemble_predictions = (predictions_1 + predictions_2 + predictions_3) / 3

    # Calcular las métricas de rendimiento
    r2 = r2_score(y_train, ensemble_predictions)
    mae = np.mean(np.abs(y_train - ensemble_predictions))

    # Registrar el ensemble en MLflow
    run_id = f"EnsembleModel_{len(mlflow.search_runs())}"
    with mlflow.start_run(run_name=run_id):
        mlflow.log_metric("mae_error", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.set_tag("model", "EnsembleModel")
        mlflow.set_tag("name", run_id)
        mlflow.set_tag("model_1", best_rf_id)
        mlflow.set_tag("model_2", best_xgb_id)
        mlflow.set_tag("model_3", best_cat_id)
    
    print("Ensemble model training completed.")


def prediction(x_test):
    print("Predicting the values...")
    best_run = mlflow.search_runs(order_by=["metrics.mae_error ASC"], max_results=1)
    best_run_id = best_run.iloc[0].run_id
    experiment_id = best_run.iloc[0].experiment_id
    print(f"Best run id: {best_run_id}, Experiment id: {experiment_id}")
    #id = mlflow.get_run(best_run_id).data.tags['name']
    #mira si el tag model es ensemble
    if mlflow.get_run(best_run_id).data.tags['model'] == 'EnsembleModel':
        best_rf_id = mlflow.get_run(best_run_id).data.tags['model_1']
        best_xgb_id = mlflow.get_run(best_run_id).data.tags['model_2']
        best_cat_id = mlflow.get_run(best_run_id).data.tags['model_3']
        model_uri_rf = f"mlflow-artifacts:/{experiment_id}/{best_rf_id}/artifacts/model"
        model_uri_xgb = f"mlflow-artifacts:/{experiment_id}/{best_xgb_id}/artifacts/model"
        model_uri_cat = f"mlflow-artifacts:/{experiment_id}/{best_cat_id}/artifacts/model"
        best_rf = mlflow.sklearn.load_model(model_uri_rf)
        best_xgb = mlflow.sklearn.load_model(model_uri_xgb)
        best_cat = mlflow.sklearn.load_model(model_uri_cat)
        predictions_1 = best_rf.predict(x_test)
        predictions_2 = best_xgb.predict(x_test)
        predictions_3 = best_cat.predict(x_test)
        ensemble_predictions = (predictions_1 + predictions_2 + predictions_3) / 3
        return ensemble_predictions
    else:
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
    
    model_1,id_1 = train_random_forest(x_train, y_train)
    model_2,id_2 = train_xgboost(x_train, y_train)
    model_3,id_3 = train_catboost(x_train, y_train)
    train_ensemble(x_train, y_train,model_1,model_2,model_3,id_1,id_2,id_3)
    train_best_ensemble(x_train,y_train)

    
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
