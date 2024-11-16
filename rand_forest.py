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
    
    # Apply log transformation to price and area-based features
    y = np.log1p(df_numeric['Listing.Price.ClosePrice'])
    area_cols = [col for col in df_numeric.columns if 'area' in col.lower()]
    for col in area_cols:
        df_numeric[col] = np.log1p(df_numeric[col])
        
    # Add polynomial features for important numeric columns
    poly_cols = ['Structure.LivingArea', 'Structure.YearBuilt', 'zhvi_value']
    for col in poly_cols:
        if col in df_numeric.columns:
            df_numeric[f'{col}_squared'] = df_numeric[col] ** 2
    
    # Apply power transform to remaining features
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X = pt.fit_transform(df_numeric.drop(['Listing.Price.ClosePrice'], axis=1))
    
    # Split data with larger training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    
    return X_train, X_test, y_train, y_test, df_numeric.drop(['Listing.Price.ClosePrice'], axis=1).columns

def create_param_grid():
    return {
        'n_estimators': [100, 300, 500],
        'max_depth': [15, 20, 25, 30],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'max_samples': [0.7, 0.8, 0.9]
    }

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Transform predictions back to original scale
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    return {
        'mse': mse,
        'r2': r2,
        'oob_score': model.oob_score_ if hasattr(model, 'oob_score_') else None
    }

def save_feature_importance(model, feature_names, run_id):
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    # Save to CSV
    feature_imp.to_csv(f'results/feature_importance_{run_id}.csv', index=False)
    
    return feature_imp

def perform_grid_search(X_train, X_test, y_train, y_test, feature_names):
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Initialize base model
    base_model = RandomForestRegressor(
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=create_param_grid(),
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='neg_mean_squared_error'
    )
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Get timestamp for run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f'results/best_model_{run_id}.joblib')
    
    # Evaluate best model
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save feature importances
    feature_imp = save_feature_importance(best_model, feature_names, run_id)
    
    # Prepare results dictionary
    results = {
        'run_id': run_id,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'metrics': metrics,
        'cv_results': grid_search.cv_results_
    }
    
    # Save results to JSON
    with open(f'results/grid_search_results_{run_id}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
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
    print(f"\nResults saved in 'results' directory with run ID: {results['run_id']}")

if __name__ == '__main__':
    main()
