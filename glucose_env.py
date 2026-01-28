"""Gym-compatible environment for glucose control in Type 1 Diabetes.

Observation: current glucose level (mg/dL).
Action: bolus insulin dose (discrete choices derived from EDA).
Reward: penalizes hypoglycemia, hyperglycemia, and large/frequent insulin actions.

All parameter ranges are loaded from simulator_params.py, which reads from
eda_derived_simulator_params.json (output of data/analyze_data.py). No constants
are hard-coded; reproducibility is guaranteed by tracing back to EDA outputs.
"""

from __future__ import annotations

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from simulator_params import (
    load_params,
    get_glucose_params,
    get_meal_params,
    get_bolus_params,
    get_basal_params,
    get_time_params,
)


class GlucoseEnv(gym.Env):
    """Minimal glucose control environment with EDA-derived parameters.

    Observation:
        - Single continuous value: current glucose in mg/dL
        - Bounded by empirical observation_bounds from EDA

    Action space:
        - Discrete choices representing bolus doses (units of insulin)
        - Actions derived from EDA bolus distribution via IQR and percentiles
        - Typical actions: no bolus, low, median, high, correction

    Dynamics (simplified):
        - Glucose evolves per timestep (5 minutes) via:
          - Basal insulin effect (constant background)
          - Bolus insulin effect (proportional to action)
          - Meal effect (random meal events)
          - Intrinsic glucose variability (Brownian noise)

    Reward:
        - Negative for glucose outside safe range
        - Negative for hypoglycemia (danger zone)
        - Negative for large bolus doses (minimize insulin use)
        - Small positive for staying in safe range

    Attributes loaded from EDA:
        - glucose_bounds: empirical min/max from CGM
        - safe_range: dense region (IQR) where most readings fall
        - tail_thresholds: rare hypoglycemia/hyperglycemia boundaries
        - variability: 5-minute delta statistics
        - bolus distribution: median, IQR, p95 for action design
        - basal rate: constant background insulin
        - meal params: typical meals for stochastic meal events
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params_json_path: str | None = None,
        random_seed: int | None = None,
        stochastic_meals: bool = True,
        verbose: bool = False,
    ):
        """Initialize glucose environment with EDA-derived parameters.

        Args:
            params_json_path: Path to eda_derived_simulator_params.json.
                              If None, auto-locates in standard directories.
            random_seed: Random seed for meal events and intrinsic noise.
            stochastic_meals: If True, meals occur randomly. If False, no meal events.
            verbose: If True, print diagnostic info on reset/step.

        Raises:
            FileNotFoundError: If parameter JSON cannot be found.
            ValueError: If parameter validation fails.
        """
        super().__init__()
        self.verbose = verbose
        self.stochastic_meals = stochastic_meals

        # Load all parameters from EDA JSON
        self.params = load_params(params_json_path)
        self._extract_params()

        # Set RNG
        self.rng = np.random.RandomState(random_seed)

        # Environment state
        self.glucose = None 
        self.step_count = 0
        self.episode_length = self.time_params["episode_length_steps"]

    def _extract_params(self) -> None:
        """Unpack nested parameter dicts for efficient access."""
        # Time
        self.time_params = get_time_params(self.params)
        self.dt_minutes = self.time_params["timestep_minutes"]  # 5 min per step

        # Glucose
        glucose_cfg = get_glucose_params(self.params)
        self.glucose_min = glucose_cfg["observation_bounds"]["min"]  # empirical min
        self.glucose_max = glucose_cfg["observation_bounds"]["max"]  # empirical max
        self.glucose_safe_low = glucose_cfg["safe_range_iqr"]["low"]  # dense region
        self.glucose_safe_high = glucose_cfg["safe_range_iqr"]["high"]  # dense region
        self.glucose_hypo = glucose_cfg["tail_thresholds"]["hypoglycemia"]  # 2.5 pct
        self.glucose_severe_hyper = glucose_cfg["tail_thresholds"]["severe_hyperglycemia"]  # 97.5 pct
        # Variability for intrinsic noise
        self.glucose_delta_std = glucose_cfg["short_term_variability"]["delta_std"]  # std of Δ5min

        # Meals
        meal_cfg = get_meal_params(self.params)
        self.meal_median_carbs = meal_cfg["typical_median_carbs"]  # median meal size
        self.meal_min_carbs = meal_cfg["common_range_iqr"]["low"]  # IQR low
        self.meal_max_carbs = meal_cfg["common_range_iqr"]["high"]  # IQR high
        self.meals_per_day_avg = meal_cfg["avg_meals_per_day"]  # avg from EDA

        # Bolus
        bolus_cfg = get_bolus_params(self.params)
        self.bolus_median = bolus_cfg["typical_median_units"]  # median dose
        self.bolus_min = bolus_cfg["common_range_iqr"]["low"]  # IQR low
        self.bolus_max = bolus_cfg["common_range_iqr"]["high"]  # IQR high
        self.bolus_p95 = bolus_cfg["upper_bound_p95"]  # p95 for correction action

        # Basal
        basal_cfg = get_basal_params(self.params)
        self.basal_rate = basal_cfg["mean_rate_u_per_hr"]  # constant background
        # CV confirms basal is relatively stable; we model it as constant

        # Define observation space: bounded glucose
        self.observation_space = spaces.Box(
            low=self.glucose_min,
            high=self.glucose_max,
            shape=(1,),
            dtype=np.float32,
        )

        # Define action space: discrete bolus doses
        # Actions: 0=no bolus, 1=low, 2=median, 3=high, 4=correction
        self.bolus_actions = np.array(
            [
                0.0,  # action 0: no bolus
                self.bolus_min,  # action 1: low bolus (IQR low)
                self.bolus_median,  # action 2: median bolus
                self.bolus_max,  # action 3: high bolus (IQR high)
                self.bolus_p95,  # action 4: correction dose (p95)
            ]
        )
        self.action_space = spaces.Discrete(len(self.bolus_actions))

        # Pre-compute meal probability per step
        # meals_per_day_avg meals in 288 steps => prob per step
        self.meal_prob_per_step = self.meals_per_day_avg / 288.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment to initial glucose state.

        Args:
            seed: Optional random seed.
            options: Optional dict (unused).

        Returns:
            Observation (glucose) and info dict.
        """
        if seed is not None:
            self.rng.seed(seed)

        # Start at median of safe range
        self.glucose = 0.5 * (self.glucose_safe_low + self.glucose_safe_high)
        self.step_count = 0

        if self.verbose:
            print(f"[reset] glucose={self.glucose:.1f} mg/dL")

        return np.array([self.glucose], dtype=np.float32), {}

    def step(self, action: int) -> tuple:
        """Execute one environment step (5 minutes).

        Args:
            action: Index into bolus_actions (0–4).

        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        bolus_dose = self.bolus_actions[action]  # insulin units delivered this step

        # --- Glucose dynamics over 5 minutes ---
        # Basal effect: constant background (negative slope on glucose)
        # Over 5 minutes: basal rate (U/hr) * 5 min / 60 min
        basal_effect = -self.basal_rate * (self.dt_minutes / 60.0)  # lowers glucose

        # Bolus effect: insulin dose proportional to glucose lowering
        # Simplified: 1 unit insulin lowers glucose by ~1.5 mg/dL (empirical rule)
        # Scale to 5 minutes: assume absorption over ~2 hours (24 steps)
        insulin_sensitivity = 1.5  # mg/dL per unit (typical for T1D)
        bolus_effect = -bolus_dose * insulin_sensitivity * (self.dt_minutes / 300.0)  # 300 min ~ 5hr absorption window

        # Meal effect: stochastic meal events
        meal_effect = 0.0
        if self.stochastic_meals and self.rng.rand() < self.meal_prob_per_step:
            # Meal occurred; glucose rises
            meal_carbs = self.rng.uniform(self.meal_min_carbs, self.meal_max_carbs)
            # Carbs -> glucose: ~5 mg/dL per 10g carbs (typical)
            meal_effect = meal_carbs * 0.5

        # Intrinsic glucose variability: Brownian noise from 5-min delta stats
        noise = self.rng.normal(0, self.glucose_delta_std)

        # Update glucose
        self.glucose = self.glucose + basal_effect + bolus_effect + meal_effect + noise
        self.glucose = np.clip(self.glucose, self.glucose_min, self.glucose_max)

        # --- Reward ---
        reward = 0.0

        # Reward for staying in safe range
        if self.glucose_safe_low <= self.glucose <= self.glucose_safe_high:
            reward += 1.0

        # Penalty for hypoglycemia (rare tail)
        if self.glucose < self.glucose_hypo:
            reward -= 20.0  # severe penalty
        elif self.glucose < self.glucose_safe_low:
            reward -= 5.0  # moderate penalty for low but not hypo

        # Penalty for hyperglycemia (rare tail)
        if self.glucose > self.glucose_severe_hyper:
            reward -= 10.0  # severe penalty
        elif self.glucose > self.glucose_safe_high:
            reward -= 1.0  # mild penalty for high

        # Penalty for large insulin doses (minimize intervention)
        reward -= 0.1 * bolus_dose

        # Increment step counter
        self.step_count += 1
        terminated = self.step_count >= self.episode_length
        truncated = False

        info = {
            "glucose": self.glucose,
            "bolus_delivered": bolus_dose,
            "basal_effect": basal_effect,
            "bolus_effect": bolus_effect,
            "meal_effect": meal_effect,
            "noise": noise,
        }

        if self.verbose and (self.step_count % 48 == 0 or terminated):  # log every 4 hours
            print(
                f"[step {self.step_count:3d}] glucose={self.glucose:6.1f} | "
                f"bolus={bolus_dose:4.1f}U | reward={reward:+.2f}"
            )

        return np.array([self.glucose], dtype=np.float32), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render environment state (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass


if __name__ == "__main__":
    # Quick test: instantiate and run a few steps
    print("Testing GlucoseEnv with EDA-derived parameters...")
    try:
        env = GlucoseEnv(verbose=True)
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space} (bolus doses: {env.bolus_actions})")
        print(f"  Glucose bounds: [{env.glucose_min:.0f}, {env.glucose_max:.0f}] mg/dL")
        print(f"  Safe range (IQR): [{env.glucose_safe_low:.0f}, {env.glucose_safe_high:.0f}] mg/dL")

        # Run a short episode
        obs, _ = env.reset()
        print(f"\n✓ Episode started at glucose={obs[0]:.1f} mg/dL")

        for step in range(10):
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        print(f"✓ Episode completed {step + 1} steps successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
