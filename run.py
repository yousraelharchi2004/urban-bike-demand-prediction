#!/usr/bin/env python3
"""
Bike Rental Demand Prediction System
Urban shared bike rental demand prediction and analysis by multimodal features
"""

# ============================================================================
# WARNING SUPPRESSION - MUST BE THE FIRST THING IN THE FILE
# ============================================================================
import warnings

# Suppress ALL sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress FutureWarnings from pandas
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# MAIN APPLICATION CODE
# ============================================================================
import os
import sys
import webbrowser
from threading import Timer

def check_dependencies():
    """Check if all required files and folders exist"""
    print("🔍 Checking project structure...")
    
    # Check required folders
    required_folders = ['app', 'app/templates', 'app/static', 'models', 'data/raw']
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"  ✅ {folder}/")
        else:
            print(f"  ❌ {folder}/ - Missing!")
            return False
    
    # Check required files
    required_files = {
        'Model file': 'models/random_forest_model.pkl',
        'Dataset': 'data/raw/seoul_bike.csv',
        'Main template': 'app/templates/index.html',
        'Backend routes': 'app/routes.py',
        'Utils': 'app/utils.py'
    }
    
    all_good = True
    for desc, filepath in required_files.items():
        if os.path.exists(filepath):
            print(f"  ✅ {desc}: {filepath}")
        else:
            print(f"  ❌ {desc}: {filepath} - Missing!")
            all_good = False
    
    return all_good

def open_browser():
    """Open browser after a short delay"""
    try:
        Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
        print("🌐 Opening browser automatically...")
    except:
        print("📱 Please open your browser manually to: http://127.0.0.1:5000")

def print_system_info():
    """Print system and project information"""
    print("\n" + "="*70)
    print("🚲 URBAN BIKE RENTAL DEMAND PREDICTION SYSTEM")
    print("🎓 Graduation Project - Multimodal Feature Analysis")
    print("="*70)
    
    print("\n" + "="*70)
    print("📊 PROJECT INFORMATION")
    print("="*70)
    print(f"📁 Project root: {os.path.abspath('.')}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    
    # Check Python packages
    print("\n📦 Checking Python packages...")
    try:
        import importlib.metadata
        flask_version = importlib.metadata.version("flask")
        print(f"  ✅ Flask: {flask_version}")
    except:
        print("  ❌ Flask not installed")
    
    try:
        import importlib.metadata
        pandas_version = importlib.metadata.version("pandas")
        print(f"  ✅ Pandas: {pandas_version}")
    except:
        print("  ❌ Pandas not installed")
    
    try:
        import importlib.metadata
        sklearn_version = importlib.metadata.version("scikit-learn")
        print(f"  ✅ Scikit-learn: {sklearn_version}")
    except:
        print("  ❌ Scikit-learn not installed")

def show_model_metrics():
    """Display model metrics without loading the model"""
    print("\n" + "="*70)
    print("🤖 MODEL INFORMATION")
    print("="*70)
    
    # Just show basic model info without loading
    print("📋 Model file exists: models/random_forest_model.pkl")
    print("📋 Expected features: 14 (from Random Forest model)")
    
    # Try to load and display metrics from the model_metrics module
    try:
        from app.model_metrics import generate_performance_report
        print(generate_performance_report())
    except ImportError as e:
        print(f"📊 Model metrics module not available: {e}")
    except Exception as e:
        print(f"📊 Could not load performance metrics: {e}")

def main():
    """Main entry point for the application"""
    
    # Check project structure
    if not check_dependencies():
        print("\n⚠️  Some files are missing. Please check the structure above.")
        print("   Make sure you've:")
        print("   1. Trained the model and saved to models/")
        print("   2. Placed dataset in data/raw/")
        print("   3. Created all template files")
        choice = input("\nContinue anyway? (y/n): ")
        if choice.lower() != 'y':
            sys.exit(1)
    
    # Print system info
    print_system_info()
    
    # Show model metrics (without loading model)
    show_model_metrics()
    
    # Create and configure the app
    try:
        from app import create_app
        app = create_app()
        
        print("\n" + "="*70)
        print("✅ APPLICATION READY")
        print("="*70)
        print("🌐 Server will start at: http://127.0.0.1:5000")
        print("\n📋 Available endpoints:")
        print("   • Homepage:        http://127.0.0.1:5000/")
        print("   • API Status:      http://127.0.0.1:5000/status")
        print("   • Model Info:      http://127.0.0.1:5000/model-info")
        print("   • Model Metrics:   http://127.0.0.1:5000/metrics")
        print("   • Prediction History: http://127.0.0.1:5000/prediction-history")
        print("   • Test Page:       http://127.0.0.1:5000/test")
        print("   • API Prediction:  http://127.0.0.1:5000/predict (POST)")
        
        print("\n🔧 Development tools:")
        print("   • Debug mode:      ON")
        print("   • Auto-reload:     ON")
        print("   • Console output:  Detailed logs")
        
        print("\n📝 Quick test command:")
        print('   curl -X POST http://127.0.0.1:5000/predict \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{')
        print('       "Hour": 14,')
        print('       "Temperature(캜)": 22.5,')
        print('       "Humidity(%)": 60,')
        print('       "Wind speed (m/s)": 2.5,')
        print('       "Visibility (10m)": 2000,')
        print('       "Dew point temperature(캜)": 10,')
        print('       "Solar Radiation (MJ/m2)": 1.5,')
        print('       "Rainfall(mm)": 0,')
        print('       "Snowfall (cm)": 0,')
        print('       "Seasons_Spring": 0,')
        print('       "Seasons_Summer": 1,')
        print('       "Seasons_Winter": 0,')
        print('       "Holiday_No Holiday": 1,')
        print('       "Functioning Day_Yes": 1')
        print('     }\'')
        
        print("\n" + "="*70)
        print("🚀 Starting Flask application...")
        print("="*70)
        print("Press Ctrl+C to stop the server\n")
        
        # Try to open browser automatically
        open_browser()
        
        # Run the app
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
        
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Check if all dependencies are installed")
        print("   2. Verify your model file exists")
        print("   3. Check app/__init__.py for syntax errors")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()