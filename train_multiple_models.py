"""
Train and compare multiple models for bike demand prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Import models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(filepath='data/raw/seoul_bike.csv'):
    """Load and preprocess the Seoul bike sharing dataset"""
    print("📊 Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(filepath, encoding='latin1')
    
    # Convert Date to datetime - FIXED: Added dayfirst=True for DD/MM/YYYY format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Extract useful features from Date
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Encode categorical variables
    categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Define features and target
    features_to_drop = ['Date', 'Rented Bike Count']
    X = df.drop(features_to_drop, axis=1)
    y = df['Rented Bike Count']
    
    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Target range: {y.min()} to {y.max()} bikes")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return X, y, label_encoders

def train_test_models(X, y):
    """Train and test multiple models"""
    print("\n" + "="*60)
    print("🤖 TRAINING MULTIPLE MODELS")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    # Scale features (optional, but helps some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with basic hyperparameters
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'─'*40}")
        print(f"Training {name}...")
        
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Calculate overfitting gap
            overfitting_gap = train_rmse - test_rmse
            
            # Store results
            results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'overfitting_gap': overfitting_gap,
                'model': model
            }
            
            trained_models[name] = model
            
            print(f"✅ {name} trained successfully!")
            print(f"   Test R²: {test_r2:.4f}")
            print(f"   Test RMSE: {test_rmse:.2f} bikes")
            print(f"   Overfitting gap: {overfitting_gap:.2f}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            traceback.print_exc()
    
    return results, trained_models, X_test_scaled, y_test, scaler

def hyperparameter_tuning(X, y, model_type='xgboost'):
    """Perform hyperparameter tuning for a specific model"""
    print(f"\n🎯 Hyperparameter Tuning for {model_type.upper()}")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grids for each model
    param_grids = {
        'xgboost': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        },
        'lightgbm': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9, -1],  # -1 means no limit
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 100, 150],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'random_forest': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    }
    
    # Select model and parameters
    if model_type == 'xgboost':
        model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        param_grid = param_grids['xgboost']
    elif model_type == 'lightgbm':
        model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        param_grid = param_grids['lightgbm']
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = param_grids['random_forest']
    else:
        print(f"❌ Model type '{model_type}' not supported for tuning")
        return None
    
    # Perform Randomized Search
    print(f"🔍 Performing RandomizedSearchCV with {len(param_grid)} parameter combinations...")
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter settings to sample
        cv=3,       # 3-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error'  # We want to minimize RMSE
    )
    
    # Fit the random search
    random_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = random_search.best_estimator_
    
    # Evaluate best model
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n✅ Best Parameters: {random_search.best_params_}")
    print(f"   Best CV Score: {-random_search.best_score_:.2f} RMSE")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f} bikes")
    
    return best_model, random_search.best_params_, test_r2, test_rmse

def compare_models(results):
    """Compare model performance and create visualizations"""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'Test R²': metrics['test_r2'],
            'Test RMSE': metrics['test_rmse'],
            'Test MAE': metrics['test_mae'],
            'Overfitting Gap': metrics['overfitting_gap']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Test R²', ascending=False)
    
    print("\n🏆 Performance Ranking (by Test R²):")
    print(df_comparison.to_string(index=False))
    
    # Save comparison to CSV
    os.makedirs('data', exist_ok=True)
    df_comparison.to_csv('data/model_comparison.csv', index=False)
    print(f"\n💾 Comparison saved to 'data/model_comparison.csv'")
    
    # Create visualizations
    create_comparison_plots(results, df_comparison)
    
    # Return best model name
    best_model = df_comparison.iloc[0]['Model']
    print(f"\n🎯 Best Performing Model: {best_model}")
    
    return best_model

def create_comparison_plots(results, df_comparison):
    """Create visualizations comparing model performance"""
    print("\n📈 Creating comparison visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. R² Comparison Bar Plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: R² Scores
    plt.subplot(2, 2, 1)
    bars = plt.bar(df_comparison['Model'], df_comparison['Test R²'], color='skyblue')
    plt.title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Subplot 2: RMSE Comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(df_comparison['Model'], df_comparison['Test RMSE'], color='lightcoral')
    plt.title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('RMSE (bikes)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Subplot 3: MAE Comparison
    plt.subplot(2, 2, 3)
    bars = plt.bar(df_comparison['Model'], df_comparison['Test MAE'], color='lightgreen')
    plt.title('Model Comparison: MAE', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('MAE (bikes)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Subplot 4: Overfitting Gap
    plt.subplot(2, 2, 4)
    positive_gap = df_comparison['Overfitting Gap'].apply(lambda x: max(x, 0))
    negative_gap = df_comparison['Overfitting Gap'].apply(lambda x: min(x, 0))
    
    plt.bar(df_comparison['Model'], positive_gap, color='orange', label='Overfitting (train < test)')
    plt.bar(df_comparison['Model'], negative_gap, color='purple', label='Underfitting (train > test)')
    plt.title('Overfitting/Underfitting Gap', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('RMSE Gap (train - test)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Visualization saved to 'data/model_comparison.png'")
    plt.show()
    
    # 2. Feature Importance for tree-based models
    print("\n🔍 Analyzing feature importance...")
    
    # Get the best performing tree-based model
    best_model_name = df_comparison.iloc[0]['Model']
    best_model_results = results[best_model_name]
    
    if hasattr(best_model_results['model'], 'feature_importances_'):
        feature_importance = best_model_results['model'].feature_importances_
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        
        # Assuming X has 17 features (original 14 + 3 new date features)
        feature_names = [
            'Hour', 'Temperature', 'Humidity', 'Wind Speed', 'Visibility',
            'Dew Point', 'Solar Radiation', 'Rainfall', 'Snowfall',
            'Seasons', 'Holiday', 'Functioning Day',
            'Month', 'DayOfWeek', 'IsWeekend', 'Year', 'Day'
        ]
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        
        plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.xticks(range(len(feature_importance)), 
                  [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(f'data/feature_importance_{best_model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"💾 Feature importance saved for {best_model_name}")
        plt.show()

def save_best_model(results, scaler, model_name=None):
    """Save the best performing model and all models individually"""
    os.makedirs('models', exist_ok=True)
    
    # Save all models individually
    print("\n💾 Saving all models...")
    for name, model_data in results.items():
        model = model_data['model']
        
        # Save individual model
        model_filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_filename)
        print(f"   Saved {name} to {model_filename}")
        
        # Save metrics for this model
        metrics = {
            'model_name': name,
            'test_r2': model_data['test_r2'],
            'test_rmse': model_data['test_rmse'],
            'test_mae': model_data['test_mae'],
            'overfitting_gap': model_data['overfitting_gap'],
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metrics_filename = f'models/{name.lower().replace(" ", "_")}_metrics.json'
        import json
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Also save the best model separately
    if model_name is None:
        # Find best model by R² score
        best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    else:
        best_model_name = model_name
    
    best_model = results[best_model_name]['model']
    
    # Create a dictionary with model and metadata
    model_package = {
        'model': best_model,
        'model_name': best_model_name,
        'test_r2': results[best_model_name]['test_r2'],
        'test_rmse': results[best_model_name]['test_rmse'],
        'scaler': scaler,
        'features_used': best_model.n_features_in_ if hasattr(best_model, 'n_features_in_') else 'Unknown',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save best model
    model_filename = f'models/best_model_{best_model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model_package, model_filename)
    
    print(f"\n💾 Best model ({best_model_name}) saved to: {model_filename}")
    print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"   Test RMSE: {results[best_model_name]['test_rmse']:.2f}")
    
    return model_filename

def main():
    """Main function to run model training and comparison"""
    print("="*60)
    print("🚲 BIKE DEMAND PREDICTION - MODEL COMPARISON")
    print("="*60)
    
    # 1. Load and preprocess data
    try:
        X, y, label_encoders = load_and_preprocess_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Trying alternative date parsing...")
        # Try alternative approach
        df = pd.read_csv('data/raw/seoul_bike.csv', encoding='latin1')
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        features_to_drop = ['Date', 'Rented Bike Count']
        X = df.drop(features_to_drop, axis=1)
        y = df['Rented Bike Count']
        
        print(f"✅ Data loaded via alternative method: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 2. Train and test multiple models
    results, trained_models, X_test, y_test, scaler = train_test_models(X, y)
    
    # 3. Compare models
    best_model_name = compare_models(results)
    
    # 4. Optional: Hyperparameter tuning for the best model
    print("\n" + "="*60)
    print("⚙️ OPTIONAL HYPERPARAMETER TUNING")
    print("="*60)
    
    tune_choice = input("\nDo you want to perform hyperparameter tuning? (y/n): ").lower()
    
    if tune_choice == 'y':
        # Map model names to tuning types
        model_mapping = {
            'Random Forest': 'random_forest',
            'XGBoost': 'xgboost',
            'LightGBM': 'lightgbm',
            'Gradient Boosting': 'xgboost'  # Use XGBoost grid for Gradient Boosting
        }
        
        tuning_type = model_mapping.get(best_model_name, 'xgboost')
        
        tuned_model, best_params, tuned_r2, tuned_rmse = hyperparameter_tuning(
            X, y, model_type=tuning_type
        )
        
        if tuned_model:
            # Update results with tuned model
            results[f'Tuned {best_model_name}'] = {
                'test_r2': tuned_r2,
                'test_rmse': tuned_rmse,
                'test_mae': mean_absolute_error(y_test, tuned_model.predict(X_test)),
                'overfitting_gap': 0,  # Would need to recalculate
                'model': tuned_model
            }
            
            print(f"\n🎯 Tuned {best_model_name} improvement:")
            print(f"   Original R²: {results[best_model_name]['test_r2']:.4f}")
            print(f"   Tuned R²: {tuned_r2:.4f}")
            print(f"   Improvement: {tuned_r2 - results[best_model_name]['test_r2']:.4f}")
    
    # 5. Save the best model and all models
    model_filename = save_best_model(results, scaler, best_model_name)
    
    print("\n" + "="*60)
    print("✅ MODEL COMPARISON COMPLETE")
    print("="*60)
    print("\n📋 Summary:")
    print(f"   • Best model: {best_model_name}")
    print(f"   • Best model saved: {model_filename}")
    print(f"   • All models saved in: models/")
    print(f"   • Comparison results saved: data/model_comparison.csv")
    print(f"   • Visualization saved: data/model_comparison.png")
    
    # Create a simple report
    os.makedirs('data', exist_ok=True)
    with open('data/model_comparison_report.txt', 'w') as f:
        f.write("BIKE DEMAND PREDICTION - MODEL COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Test R²: {metrics['test_r2']:.4f}\n")
            f.write(f"  Test RMSE: {metrics['test_rmse']:.2f}\n")
            f.write(f"  Test MAE: {metrics['test_mae']:.2f}\n")
            f.write(f"  Overfitting Gap: {metrics['overfitting_gap']:.2f}\n\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"RECOMMENDATION: Use {best_model_name} for production\n")
        f.write(f"\nModels saved:\n")
        for name in results.keys():
            f.write(f"- {name.lower().replace(' ', '_')}_model.pkl\n")
            f.write(f"- {name.lower().replace(' ', '_')}_metrics.json\n")
    
    print(f"💾 Report saved: data/model_comparison_report.txt")
    
    # Show where to find everything
    print("\n📁 Files created:")
    print("   data/model_comparison.csv")
    print("   data/model_comparison.png")
    print("   data/model_comparison_report.txt")
    print("   models/random_forest_model.pkl")
    print("   models/xgboost_model.pkl")
    print("   models/lightgbm_model.pkl")
    print("   models/gradient_boosting_model.pkl")
    print("   models/best_model_*.pkl")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()