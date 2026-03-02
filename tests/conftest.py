"""
Shared fixtures for all tests.
"""

import os
import sys
import json
import pytest
import warnings

# Suppress sklearn warnings during tests
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def app():
    """Create Flask test app."""
    from app import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_prediction_input():
    """Sample input data for prediction tests."""
    return {
        "Hour": 14,
        "Temperature(\u00b0C)": 22.5,
        "Humidity(%)": 60,
        "Wind speed (m/s)": 2.5,
        "Visibility (10m)": 2000,
        "Dew point temperature(\u00b0C)": 10,
        "Solar Radiation (MJ/m2)": 1.5,
        "Rainfall(mm)": 0,
        "Snowfall (cm)": 0,
        "Seasons_Spring": 0,
        "Seasons_Summer": 1,
        "Seasons_Winter": 0,
        "Holiday_No Holiday": 1,
        "Functioning Day_Yes": 1,
    }


@pytest.fixture
def sample_prediction_input_korean():
    """Sample input with Korean character field names (original model format)."""
    return {
        "Hour": 14,
        "Temperature(\uce9c)": 22.5,
        "Humidity(%)": 60,
        "Wind speed (m/s)": 2.5,
        "Visibility (10m)": 2000,
        "Dew point temperature(\uce9c)": 10,
        "Solar Radiation (MJ/m2)": 1.5,
        "Rainfall(mm)": 0,
        "Snowfall (cm)": 0,
        "Seasons_Spring": 0,
        "Seasons_Summer": 1,
        "Seasons_Winter": 0,
        "Holiday_No Holiday": 1,
        "Functioning Day_Yes": 1,
    }


@pytest.fixture
def sample_advanced_input():
    """Sample input for advanced/xgboost model format."""
    return {
        "hour": 14,
        "temperature": 22.5,
        "humidity": 60,
        "wind_speed": 2.5,
        "visibility": 2000,
        "dew_point": 10,
        "solar_radiation": 1.5,
        "rainfall": 0,
        "snowfall": 0,
        "day_of_week": 3,
        "season_Summer": 1,
        "season_Spring": 0,
        "season_Winter": 0,
        "season_Autumn": 0,
        "holiday_No Holiday": 1,
        "functioning_day_Yes": 1,
    }
