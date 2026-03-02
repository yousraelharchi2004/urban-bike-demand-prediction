# app/config.py
import warnings
import sklearn

def configure_warnings():
    """Configure warning settings for the application"""
    
    # Ignore the specific sklearn unpickling warning
    warnings.filterwarnings('ignore', 
                          category=UserWarning,
                          module='sklearn',
                          message='Trying to unpickle estimator')
    
    # Ignore pandas future warnings
    warnings.filterwarnings('ignore', 
                          category=FutureWarning,
                          module='pandas')
    
    # Show all other warnings
    warnings.simplefilter('always', category=DeprecationWarning)
    
    print(f" Scikit-learn version: {sklearn.__version__}")
    print("  Configured to ignore sklearn version mismatch warnings")