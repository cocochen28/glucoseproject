"""Load and expose data-driven simulator parameters derived from OhioT1DM EDA.

This module loads parameters from 'eda_derived_simulator_params.json', validates them,
and provides accessor functions for different parameter categories. All numeric values
come directly from the EDA analysis; only timestep and episode length are fixed
based on the CGM observation frequency (5-minute intervals).

Usage:
    from simulator_params import load_params, get_glucose_params, get_meal_params

    params = load_params()
    glucose_cfg = get_glucose_params(params)
    meal_cfg = get_meal_params(params)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


# Timestep and episode length are fixed based on CGM sampling rate (5-minute intervals)
# 288 steps = 24 hours * 60 minutes / 5 minutes per step (one full day episode)
TIMESTEP_MINUTES = 5  # CGM reporting interval in the OhioT1DM dataset
EPISODE_LENGTH_STEPS = 288  # 24 hours of 5-minute readings


def _find_params_file() -> Path:
    """Locate eda_derived_simulator_params.json in standard locations.

    Searches in:
      1. ./data/plots_eda/
      2. ./plots_eda/
      3. Current directory
    """
    candidates = [
        Path("./data/plots_eda/eda_derived_simulator_params.json"),
        Path("./plots_eda/eda_derived_simulator_params.json"),
        Path("./eda_derived_simulator_params.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find eda_derived_simulator_params.json in standard locations. "
        f"Run 'python data/analyze_data.py --data-dir data/raw/ohio_t1dm' first."
    )


def load_params(json_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load simulator parameters from the EDA-derived JSON file.

    Args:
        json_path: Optional path to JSON file. If None, searches standard locations.

    Returns:
        Dictionary with top-level keys: 'glucose', 'meals', 'bolus', 'basal', 'validation'.

    Raises:
        FileNotFoundError: If the JSON file cannot be found.
        json.JSONDecodeError: If the JSON is malformed.
        ValueError: If required keys are missing or validation fails.
    """
    if json_path is None:
        json_path = _find_params_file()
    else:
        json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {json_path}")

    with open(json_path, "r") as f:
        params = json.load(f)

    # Validate top-level structure
    required_keys = {"glucose", "meals", "bolus", "basal"}
    if not required_keys.issubset(set(params.keys())):
        missing = required_keys - set(params.keys())
        raise ValueError(f"Missing required parameter sections: {missing}")

    # Validate glucose section
    glucose = params.get("glucose", {})
    if not {"observation_bounds", "short_term_variability", "safe_range_iqr", "tail_thresholds"}.issubset(
        set(glucose.keys())
    ):
        raise ValueError("Incomplete 'glucose' section in parameters")

    # Validate that observation bounds make sense (min < max)
    obs = glucose["observation_bounds"]
    if obs["min"] >= obs["max"]:
        raise ValueError(
            f"Invalid glucose observation bounds: min={obs['min']} >= max={obs['max']}"
        )

    # Validate meals section
    meals = params.get("meals", {})
    if not {"typical_median_carbs", "common_range_iqr", "upper_cap_p95", "avg_meals_per_day"}.issubset(
        set(meals.keys())
    ):
        raise ValueError("Incomplete 'meals' section in parameters")

    # Validate bolus section
    bolus = params.get("bolus", {})
    if not {"typical_median_units", "common_range_iqr", "upper_bound_p95"}.issubset(set(bolus.keys())):
        raise ValueError("Incomplete 'bolus' section in parameters")

    # Validate basal section
    basal = params.get("basal", {})
    if not {"mean_rate_u_per_hr", "typical_range_iqr", "coefficient_of_variation"}.issubset(set(basal.keys())):
        raise ValueError("Incomplete 'basal' section in parameters")

    return params


def get_time_params(params: Dict[str, Any]) -> Dict[str, int]:
    """Return simulator time parameters aligned with CGM sampling.

    Returns:
        Dictionary with keys:
          - 'timestep_minutes': 5 (CGM reporting interval)
          - 'episode_length_steps': 288 (one full day)

    Rationale:
        OhioT1DM data uses 5-minute CGM intervals. 288 steps = 24 hours.
        These are fixed based on the dataset, not EDA-derived.
    """
    return {
        "timestep_minutes": TIMESTEP_MINUTES,
        "episode_length_steps": EPISODE_LENGTH_STEPS,
    }


def get_glucose_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract glucose dynamics parameters from loaded config.

    Returns:
        Dictionary with keys:
          - 'observation_bounds': {'min', 'max'} — empirical extremes from CGM
          - 'short_term_variability': {'delta_std', 'abs_delta_p95'} — from 5-minute deltas
          - 'safe_range_iqr': {'low', 'high'} — 25th–75th percentile (dense region)
          - 'tail_thresholds': {'hypoglycemia', 'severe_hyperglycemia'} — 2.5th and 97.5th pct

    Derivation:
        - observation_bounds: min/max of all CGM readings
        - short_term_variability:
          - delta_std: std of consecutive 5-minute differences
          - abs_delta_p95: 95th percentile of |delta| (typical swings + occasional jumps)
        - safe_range_iqr: IQR captures typical glucose distribution
        - tail_thresholds: rare tails from percentiles (data-driven, not clinical)
    """
    glucose = params["glucose"]
    return {
        "observation_bounds": glucose["observation_bounds"],
        "short_term_variability": glucose["short_term_variability"],
        "safe_range_iqr": glucose["safe_range_iqr"],
        "tail_thresholds": glucose["tail_thresholds"],
    }


def get_meal_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract meal parameters from loaded config.

    Returns:
        Dictionary with keys:
          - 'typical_median_carbs': median carbohydrate amount per meal
          - 'common_range_iqr': {'low', 'high'} — 25th–75th percentile
          - 'upper_cap_p95': 95th percentile (excludes rare outliers)
          - 'avg_meals_per_day': mean meals per day (per-patient average)

    Derivation:
        - typical_median_carbs: median of all observed meal carbs
        - common_range_iqr: IQR captures typical meal sizes
        - upper_cap_p95: excludes extreme outliers without hard-coding
        - avg_meals_per_day: mean of per-patient daily meal counts
    """
    meals = params["meals"]
    return {
        "typical_median_carbs": meals["typical_median_carbs"],
        "common_range_iqr": meals["common_range_iqr"],
        "upper_cap_p95": meals["upper_cap_p95"],
        "avg_meals_per_day": meals["avg_meals_per_day"],
    }


def get_bolus_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract bolus insulin parameters from loaded config.

    Returns:
        Dictionary with keys:
          - 'typical_median_units': median bolus dose in units
          - 'common_range_iqr': {'low', 'high'} — 25th–75th percentile
          - 'upper_bound_p95': 95th percentile (excludes rare extreme corrections)

    Derivation:
        - typical_median_units: median of all observed bolus doses
        - common_range_iqr: IQR captures typical dosing scale
        - upper_bound_p95: excludes rare extreme corrections without hard-coding
    """
    bolus = params["bolus"]
    return {
        "typical_median_units": bolus["typical_median_units"],
        "common_range_iqr": bolus["common_range_iqr"],
        "upper_bound_p95": bolus["upper_bound_p95"],
    }


def get_basal_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract basal insulin parameters from loaded config.

    Returns:
        Dictionary with keys:
          - 'mean_rate_u_per_hr': mean basal rate across patients
          - 'typical_range_iqr': {'low', 'high'} — 25th–75th percentile
          - 'coefficient_of_variation': std/mean (low CV justifies constant background modeling)

    Derivation:
        - mean_rate_u_per_hr: mean of all observed basal rates
        - typical_range_iqr: IQR of observed basal rates
        - coefficient_of_variation: CV = std/mean; low CV (~0.26) indicates basal acts
          as a relatively constant background, not a primary control variable
    """
    basal = params["basal"]
    return {
        "mean_rate_u_per_hr": basal["mean_rate_u_per_hr"],
        "typical_range_iqr": basal["typical_range_iqr"],
        "coefficient_of_variation": basal["coefficient_of_variation"],
    }


if __name__ == "__main__":
    # Quick test: load and print all parameters
    try:
        params = load_params()
        print("✓ Successfully loaded EDA-derived simulator parameters")
        print(f"✓ Glucose bounds: {get_glucose_params(params)['observation_bounds']}")
        print(f"✓ Time params: {get_time_params(params)}")
        print(f"✓ Meals: median={get_meal_params(params)['typical_median_carbs']:.1f}g, "
              f"range={get_meal_params(params)['common_range_iqr']}")
        print(f"✓ Bolus: median={get_bolus_params(params)['typical_median_units']:.1f}U, "
              f"range={get_bolus_params(params)['common_range_iqr']}")
        print(f"✓ Basal: mean={get_basal_params(params)['mean_rate_u_per_hr']:.2f}U/hr, "
              f"CV={get_basal_params(params)['coefficient_of_variation']:.3f}")
    except Exception as e:
        print(f"✗ Error loading parameters: {e}")
        exit(1)
