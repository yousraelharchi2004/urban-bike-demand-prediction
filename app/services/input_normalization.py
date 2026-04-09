"""
Input normalization helpers extracted from the legacy routes module.
"""

from __future__ import annotations


def _find_key(data: dict, *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in data:
            return candidate
    return None


def _find_temperature_key(data: dict) -> str | None:
    for key in data.keys():
        lowered = str(key).lower()
        if 'temperature' in lowered and 'dew' not in lowered:
            return key
    return None


def _find_dew_point_key(data: dict) -> str | None:
    for key in data.keys():
        lowered = str(key).lower()
        if 'dew point temperature' in lowered:
            return key
    return None


def _float_value(data: dict, *keys: str, default: float) -> float:
    key = _find_key(data, *keys)
    if key is None:
        return float(default)
    try:
        return float(data[key])
    except (TypeError, ValueError):
        return float(default)


def _season_encoding(data: dict) -> dict[str, float]:
    spring = _float_value(data, 'Seasons_Spring', 'season_Spring', default=0)
    summer = _float_value(data, 'Seasons_Summer', 'season_Summer', default=0)
    winter = _float_value(data, 'Seasons_Winter', 'season_Winter', default=0)
    autumn = _float_value(data, 'Seasons_Autumn', 'season_Autumn', default=0)
    if spring == 0 and summer == 0 and winter == 0 and autumn == 0:
        summer = 1.0
    return {
        'season_Spring': spring,
        'season_Summer': summer,
        'season_Winter': winter,
        'season_Autumn': autumn,
    }


def _holiday_encoding(data: dict) -> dict[str, float]:
    no_holiday = _find_key(data, 'Holiday_No Holiday', 'holiday_No Holiday')
    holiday = _find_key(data, 'Holiday_Holiday', 'holiday_Holiday')
    if no_holiday is not None:
        no_holiday_value = _float_value(data, no_holiday, default=1)
        holiday_value = 1 - no_holiday_value
    elif holiday is not None:
        holiday_value = _float_value(data, holiday, default=0)
        no_holiday_value = 1 - holiday_value
    else:
        holiday_value = 0.0
        no_holiday_value = 1.0
    return {
        'holiday_Holiday': holiday_value,
        'holiday_No Holiday': no_holiday_value,
    }


def _functioning_day_encoding(data: dict) -> dict[str, float]:
    yes_key = _find_key(data, 'Functioning Day_Yes', 'functioning_day_Yes')
    no_key = _find_key(data, 'Functioning Day_No', 'functioning_day_No')
    if yes_key is not None:
        yes_value = _float_value(data, yes_key, default=1)
        no_value = 1 - yes_value
    elif no_key is not None:
        no_value = _float_value(data, no_key, default=0)
        yes_value = 1 - no_value
    else:
        yes_value = 1.0
        no_value = 0.0
    return {
        'functioning_day_Yes': yes_value,
        'functioning_day_No': no_value,
    }


def _normalize_advanced_family(data: dict) -> dict:
    temperature_key = _find_temperature_key(data)
    dew_point_key = _find_dew_point_key(data)

    normalized = {
        'hour': _float_value(data, 'Hour', 'hour', default=12),
        'temperature': _float_value(data, *(filter(None, [temperature_key, 'Temperature(°C)', 'Temperature(Â°C)', 'temperature'])), default=20),
        'Humidity%': _float_value(data, 'Humidity(%)', 'humidity', 'Humidity%', default=50),
        'Wind_speed_m/s': _float_value(data, 'Wind speed (m/s)', 'wind_speed', 'Wind_speed_m/s', default=2.5),
        'visibility': _float_value(data, 'Visibility (10m)', 'visibility', default=2000),
        'dew_point': _float_value(data, *(filter(None, [dew_point_key, 'Dew point temperature(°C)', 'Dew point temperature(Â°C)', 'dew_point'])), default=10),
        'Solar_Radiation_MJ/m2': _float_value(data, 'Solar Radiation (MJ/m2)', 'solar_radiation', 'Solar_Radiation_MJ/m2', default=1.5),
        'rainfall': _float_value(data, 'Rainfall(mm)', 'rainfall', default=0),
        'snowfall': _float_value(data, 'Snowfall (cm)', 'snowfall', default=0),
        'day_of_week': _float_value(data, 'day_of_week', default=3),
    }
    normalized.update(_season_encoding(data))
    normalized.update(_holiday_encoding(data))
    normalized.update(_functioning_day_encoding(data))
    return normalized


def normalize_field_names_for_model(data: dict, target_model: str) -> dict:
    """
    Normalize incoming payload keys to the names expected by each model family.
    """
    if target_model == 'original':
        return data

    if target_model in ('advanced', 'xgboost', 'lightgbm', 'stacking'):
        return _normalize_advanced_family(data)

    if target_model == 'improved':
        normalized = dict(data)
        temperature_key = _find_temperature_key(data)
        dew_point_key = _find_dew_point_key(data)
        if temperature_key is not None:
            normalized['Temperature(Â°C)'] = _float_value(data, temperature_key, default=20)
        if dew_point_key is not None:
            normalized['Dew point temperature(Â°C)'] = _float_value(data, dew_point_key, default=10)
        return normalized

    return dict(data)


def get_required_fields_for_model(model_type: str) -> list[str]:
    """Return the base required inputs for a specific model family."""
    if model_type == 'original':
        return [
            'Hour', 'Temperature(ìºœ)', 'Humidity(%)', 'Wind speed (m/s)',
            'Visibility (10m)', 'Dew point temperature(ìºœ)',
            'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)',
            'Seasons_Spring', 'Seasons_Summer', 'Seasons_Winter',
            'Holiday_No Holiday', 'Functioning Day_Yes'
        ]
    if model_type == 'improved':
        return ['Hour', 'day_of_week']
    if model_type in ('advanced', 'xgboost', 'lightgbm', 'stacking'):
        return ['hour', 'temperature', 'day_of_week']
    return []
