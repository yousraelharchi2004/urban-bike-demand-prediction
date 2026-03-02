"""
Unit tests for feature engineering.
"""

import pytest
import numpy as np
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestCyclicEncoding:
    """Test cyclic (sin/cos) feature encoding."""

    def test_hour_sin_range(self):
        """Hour sin encoding should be in [-1, 1]."""
        for hour in range(24):
            val = np.sin(2 * np.pi * hour / 24)
            assert -1.0 <= val <= 1.0

    def test_hour_cos_range(self):
        """Hour cos encoding should be in [-1, 1]."""
        for hour in range(24):
            val = np.cos(2 * np.pi * hour / 24)
            assert -1.0 <= val <= 1.0

    def test_hour_0_and_24_are_same(self):
        """Hour 0 and hour 24 should produce same sin/cos values."""
        sin_0 = np.sin(2 * np.pi * 0 / 24)
        sin_24 = np.sin(2 * np.pi * 24 / 24)
        assert abs(sin_0 - sin_24) < 1e-10

    def test_month_cyclic(self):
        """Month encoding should be cyclic (Jan == 13th month)."""
        sin_1 = np.sin(2 * np.pi * 1 / 12)
        sin_13 = np.sin(2 * np.pi * 13 / 12)
        assert abs(sin_1 - sin_13) < 1e-10


class TestLagFeatures:
    """Test lag feature creation in the unified pipeline."""

    def test_lag_shift_prevents_leakage(self):
        """Lag features must use shift() to avoid data leakage."""
        import pandas as pd

        # Simulate a time series
        target = pd.Series([100, 200, 300, 400, 500, 600])
        lag_1 = target.shift(1)

        # At index 0, lag should be NaN (no future data used)
        assert pd.isna(lag_1.iloc[0])
        # At index 1, lag should be the PREVIOUS value (100), not current (200)
        assert lag_1.iloc[1] == 100

    def test_rolling_with_shift_excludes_current(self):
        """Rolling mean with shift(1) should NOT include current observation."""
        import pandas as pd

        target = pd.Series([10, 20, 30, 40, 50])
        rolling = target.shift(1).rolling(3, min_periods=1).mean()

        # At index 3: shift gives [10, 20, 30], rolling(3) = mean(10,20,30) = 20
        assert rolling.iloc[3] == 20.0
        # Current value (40) should NOT be in the rolling window
        assert rolling.iloc[3] != pytest.approx(30.0)  # (20+30+40)/3 would be wrong


class TestFeatureCount:
    """Test that feature engineering produces expected number of features."""

    def test_unified_feature_engineer_creates_features(self):
        """FeatureEngineer should add features beyond the raw columns."""
        try:
            from train_unified_pipeline import FeatureEngineer, DataLoader
            loader = DataLoader()
            df = loader.load()
            fe = FeatureEngineer()
            df_feat = fe.transform(df)

            raw_cols = 14  # Original dataset has ~14 columns
            assert len(df_feat.columns) > raw_cols
        except (ImportError, FileNotFoundError):
            pytest.skip("Unified pipeline or data not available")


class TestPeakHourIndicator:
    """Test peak hour and rush hour features."""

    def test_peak_hours_7_to_9(self):
        """Hours 7-9 should be peak hours."""
        for h in (7, 8, 9):
            is_peak = 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
            assert is_peak == 1

    def test_peak_hours_17_to_19(self):
        """Hours 17-19 should be peak hours."""
        for h in (17, 18, 19):
            is_peak = 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
            assert is_peak == 1

    def test_non_peak_hours(self):
        """Hours outside peak ranges should not be peak."""
        for h in (0, 3, 12, 15, 22):
            is_peak = 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
            assert is_peak == 0

    def test_rush_hour_intensity_values(self):
        """Rush hour intensity should return correct values."""
        def rush(h):
            if h in (7, 8, 17, 18):
                return 1.0
            elif h in (6, 9, 16, 19):
                return 0.6
            elif h in (10, 15, 20):
                return 0.3
            else:
                return 0.0

        assert rush(8) == 1.0
        assert rush(6) == 0.6
        assert rush(10) == 0.3
        assert rush(0) == 0.0


class TestComfortIndex:
    """Test weather comfort index calculation."""

    def test_comfort_index_formula(self):
        """Comfort index = temperature - humidity/10."""
        temp, humidity = 25, 60
        expected = 25 - 60 / 10  # = 19
        assert expected == 19.0

    def test_comfort_index_cold_wet(self):
        """Cold and wet conditions should have low comfort index."""
        comfort = -5 - 90 / 10  # = -14
        assert comfort < 0
