import json
try:
    with open('models/improved_model_metrics.json', 'r') as f:
        metrics = json.load(f)
    print('📊 IMPROVED MODEL METRICS:')
    print(f'   RMSE: {metrics.get("rmse", "Not available")}')
    print(f'   R² Score: {metrics.get("r2", "Not available")}')
    print(f'   MAE: {metrics.get("mae", "Not available")}')
    print(f'   MAPE: {metrics.get("mape", "Not available")}%')
    print(f'   Number of features: {metrics.get("n_features", "Not available")}')
except FileNotFoundError:
    print('❌ Metrics file not found at models/improved_model_metrics.json')
except Exception as e:
    print(f'❌ Error: {e}')