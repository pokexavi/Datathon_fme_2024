

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from datetime import datetime
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

def prepare_data():
    # Load data
    df = pd.read_csv('train_with_zhvi.csv', low_memory=False)
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    # Remove columns that likely don't help predict price
    cols_to_drop = ['Listing.ListingId', 'Location.Address.PostalCodePlus4', 'Location.Address.CensusBlock']
    df_numeric = df_numeric.drop(columns=[col for col in cols_to_drop if col in df_numeric.columns])
    
    # Fill NaN values with median instead of mean to be more robust to outliers
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    # Remove price outliers using IQR method with tighter bounds
    Q1 = df_numeric['Listing.Price.ClosePrice'].quantile(0.25)
    Q3 = df_numeric['Listing.Price.ClosePrice'].quantile(0.75)
    IQR = Q3 - Q1
    price_lower = Q1 - 1.25 * IQR  # Reduced from 1.5 to 1.25 for tighter bounds
    price_upper = Q3 + 1.25 * IQR
    df_numeric = df_numeric[
        (df_numeric['Listing.Price.ClosePrice'] >= price_lower) & 
        (df_numeric['Listing.Price.ClosePrice'] <= price_upper)
    ]
    
    # Remove features with high correlation to avoid multicollinearity
    corr_matrix = df_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    df_numeric = df_numeric.drop(columns=to_drop)
    
    # Apply log transformation to price and area-based features in parallel
    y = np.log1p(df_numeric['Listing.Price.ClosePrice'])
    area_cols = [col for col in df_numeric.columns if 'area' in col.lower()]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda x: np.log1p(df_numeric[x]), col) for col in area_cols]
        for col, future in zip(area_cols, futures):
            df_numeric[col] = future.result()
        
    # Add polynomial features for important numeric columns in parallel
    poly_cols = ['Structure.LivingArea', 'Structure.YearBuilt', 'zhvi_value']
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda x: df_numeric[x] ** 2 if x in df_numeric.columns else None, col) for col in poly_cols]
        for col, future in zip(poly_cols, futures):
            if future.result() is not None:
                df_numeric[f'{col}_squared'] = future.result()
    
    # Apply power transform to remaining features
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X = pt.fit_transform(df_numeric.drop(['Listing.Price.ClosePrice'], axis=1))
    
    # Split data with larger training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    
    return X_train, X_test, y_train, y_test, df_numeric.drop(['Listing.Price.ClosePrice'], axis=1).columns

def create_param_grid():
    return {
        'n_estimators': [100, 300, 500],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Transform predictions back to original scale
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    # Calculate metrics in parallel
    with ThreadPoolExecutor() as executor:
        mse_future = executor.submit(mean_squared_error, y_test_orig, y_pred_orig)
        r2_future = executor.submit(r2_score, y_test_orig, y_pred_orig)
        mae_future = executor.submit(mean_squared_error, y_test_orig, y_pred_orig, squared=False)
        
        mse = mse_future.result()
        rmse = np.sqrt(mse)
        mae = mae_future.result()
        r2 = r2_future.result()
        
    # Calculate additional metrics
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
    residuals = y_test_orig - y_pred_orig
    explained_variance = 1 - np.var(residuals) / np.var(y_test_orig)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'explained_variance': explained_variance,
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals),
        'oob_score': model.oob_score_ if hasattr(model, 'oob_score_') else None
    }

def save_feature_importance(model, feature_names, run_id):
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    # Log feature importances as artifact
    feature_imp.to_csv('feature_importances.csv', index=False)
    mlflow.log_artifact('feature_importances.csv')
    
    return feature_imp

def perform_grid_search(X_train, X_test, y_train, y_test, feature_names):
    mlflow.set_experiment('housing_price_prediction')
    
    # Initialize base model
    base_model = RandomForestRegressor(
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=mp.cpu_count() # Use all available CPU cores
    )
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=create_param_grid(),
        cv=5,
        n_jobs=mp.cpu_count(), # Use all available CPU cores
        verbose=2,
        scoring='neg_mean_squared_error'
    )
    
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params(create_param_grid())
        
        # Perform grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluate best model
        metrics = evaluate_model(best_model, X_test, y_test)
        
        # Log metrics in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            metrics_to_log = {
                'mse': metrics['mse'],
                'r2': metrics['r2'],
                'best_cv_score': grid_search.best_score_
            }
            
            for metric_name, metric_value in metrics_to_log.items():
                futures.append(executor.submit(mlflow.log_metric, metric_name, metric_value))
                
            if metrics['oob_score']:
                futures.append(executor.submit(mlflow.log_metric, 'oob_score', metrics['oob_score']))
                
            # Wait for all logging to complete
            [f.result() for f in futures]
        
        # Save feature importances
        feature_imp = save_feature_importance(best_model, feature_names, None)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Log grid search results
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'metrics': metrics,
            'cv_results': grid_search.cv_results_
        }
        with open('grid_search_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        mlflow.log_artifact('grid_search_results.json')
        
        return best_model, results, feature_imp

def main():
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    
    # Perform grid search and get results
    best_model, results, feature_imp = perform_grid_search(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Print summary
    print("\nGrid Search Results Summary:")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best Cross-validation Score: {results['best_score']}")
    print(f"Test Set MSE: {results['metrics']['mse']}")
    print(f"Test Set R2: {results['metrics']['r2']}")
    print(f"Out-of-bag Score: {results['metrics']['oob_score']}")
    print("\nTop 15 Most Important Features:")
    print(feature_imp.head(15))
    print("\nResults and model artifacts have been logged to MLflow")

if __name__ == '__main__':
    mp.freeze_support() # Required for Windows
    main()