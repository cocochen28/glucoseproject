"""Gym-compatible environment for glucose control in Type 1 Diabetes.

Observation: 5-dimensional state with glucose, trend, time, bolus history.
Action: bolus insulin dose (discrete choices derived from EDA).
Reward: penalizes hypoglycemia, hyperglycemia, and large insulin doses.

All parameters loaded from simulator_params.py (EDA-derived, no hard-coded constants).

TEMPORAL DYNAMICS (v2.0):
  Realistic absorption curves replace instant impulses:
  - Meals: glucose rise distributed uniformly over 24 steps (2 hours)
  - Insulin: glucose drop distributed triangularly over 36 steps (peak effect at ~1 hour)
  - Basal: continuous effect per step (unchanged)
  - Noise: stochastic per step (unchanged)

Metrics tracked per step in info dict:
  - hypo_step: 1 if glucose < 66 mg/dL this step, else 0
  - hypo_event: 1 if we just ENTERED hypo zone, else 0
  - severe_hyper_step: 1 if glucose > 292 mg/dL this step, else 0
  - severe_hyper_event: 1 if we just ENTERED severe hyper zone, else 0
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
    """Glucose control environment with EDA-derived parameters and event tracking."""

    metadata = {"render_modes": []}

    # Reward targets are aligned with reported TIR metric (80-180 mg/dL).
    TIR_LOW = 80.0
    TIR_HIGH = 180.0

    # Zone reward magnitudes.
    REWARD_IN_RANGE = 1.0
    PENALTY_LOW = -5.0
    PENALTY_HYPO = -20.0
    PENALTY_HIGH = -2.0
    PENALTY_SEVERE_HYPER = -10.0

    # Temporal dynamics: absorption horizons (in steps of 5 minutes)
    MEAL_ABSORPTION_STEPS = 24  # 2 hours: glucose rise spreads over this many steps
    INSULIN_ABSORPTION_STEPS = 36  # 3 hours: bolus effect spreads over this many steps

    def __init__(
        self,
        params_json_path: str | None = None,
        random_seed: int | None = None,
        stochastic_meals: bool = True,
        insulin_penalty_coeff: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.stochastic_meals = stochastic_meals
        self.insulin_penalty_coeff = insulin_penalty_coeff

        # Load parameters from EDA
        self.params = load_params(params_json_path)
        self._extract_params()

        # RNG
        self.rng = np.random.RandomState(random_seed)

        # State tracking
        self.glucose = None
        self.glucose_prev = None
        self.step_count = 0
        self.last_bolus_dose = 0.0
        self.steps_since_last_bolus = 0
        self.was_in_hypo = False
        self.was_in_severe_hyper = False

        # Temporal dynamics: absorption queues
        # pending_meal_glucose: list of glucose contributions to apply over next steps (uniform distribution)
        # pending_bolus_units: list of insulin units to apply over next steps (triangular distribution)
        self.pending_meal_glucose = []
        self.pending_bolus_units = []

    def _extract_params(self) -> None:
        """Unpack parameters from EDA JSON."""
        # Time
        self.time_params = get_time_params(self.params)
        self.dt_minutes = self.time_params["timestep_minutes"]
        self.episode_length = self.time_params["episode_length_steps"]

        # Glucose bounds and thresholds
        glucose_cfg = get_glucose_params(self.params)
        self.glucose_min = glucose_cfg["observation_bounds"]["min"]
        self.glucose_max = glucose_cfg["observation_bounds"]["max"]
        self.glucose_safe_low = glucose_cfg["safe_range_iqr"]["low"]
        self.glucose_safe_high = glucose_cfg["safe_range_iqr"]["high"]
        self.glucose_hypo = glucose_cfg["tail_thresholds"]["hypoglycemia"]
        self.glucose_severe_hyper = glucose_cfg["tail_thresholds"]["severe_hyperglycemia"]
        self.glucose_delta_std = glucose_cfg["short_term_variability"]["delta_std"]

        # Meals
        meal_cfg = get_meal_params(self.params)
        self.meal_median_carbs = meal_cfg["typical_median_carbs"]
        self.meal_min_carbs = meal_cfg["common_range_iqr"]["low"]
        self.meal_max_carbs = meal_cfg["common_range_iqr"]["high"]
        self.meals_per_day_avg = meal_cfg["avg_meals_per_day"]

        # Bolus
        bolus_cfg = get_bolus_params(self.params)
        self.bolus_median = bolus_cfg["typical_median_units"]
        self.bolus_min = bolus_cfg["common_range_iqr"]["low"]
        self.bolus_max = bolus_cfg["common_range_iqr"]["high"]
        self.bolus_p95 = bolus_cfg["upper_bound_p95"]

        # Basal
        basal_cfg = get_basal_params(self.params)
        self.basal_rate = basal_cfg["mean_rate_u_per_hr"]

        # Observation space: 5-dim [glucose, trend, time_of_day, last_bolus, steps_since_bolus]
        self.observation_space = spaces.Box(
            low=np.array([self.glucose_min, -np.inf, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.glucose_max, np.inf, 1.0, self.bolus_p95, self.episode_length], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )

        # Action space: 5 discrete bolus doses
        self.bolus_actions = np.array([0.0, self.bolus_min, self.bolus_median, self.bolus_max, self.bolus_p95])
        self.action_space = spaces.Discrete(len(self.bolus_actions))

        # Meal probability per step
        self.meal_prob_per_step = self.meals_per_day_avg / 288.0

    def _get_observation(self) -> np.ndarray:
        """Build 5-dim observation vector."""
        glucose_trend = self.glucose - self.glucose_prev
        time_of_day = self.step_count / self.episode_length
        steps_since_bolus_clipped = np.clip(self.steps_since_last_bolus, 0, self.episode_length)
        
        obs = np.array([
            self.glucose,
            glucose_trend,
            time_of_day,
            self.last_bolus_dose,
            steps_since_bolus_clipped,
        ], dtype=np.float32)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset to initial state."""
        if seed is not None:
            self.rng.seed(seed)

        # Initialize at median of safe range
        self.glucose = 0.5 * (self.glucose_safe_low + self.glucose_safe_high)
        self.glucose_prev = self.glucose
        self.step_count = 0
        self.last_bolus_dose = 0.0
        self.steps_since_last_bolus = 0
        self.was_in_hypo = False
        self.was_in_severe_hyper = False

        # Clear absorption queues
        self.pending_meal_glucose = []
        self.pending_bolus_units = []

        if self.verbose:
            print(f"[reset] glucose={self.glucose:.1f} mg/dL")

        obs = self._get_observation()
        return obs, {}

    def _create_meal_absorption_profile(self, total_meal_glucose: float) -> list:
        """Create uniform meal absorption over MEAL_ABSORPTION_STEPS.
        
        Returns list of glucose increments, one per step.
        Example: total_meal_glucose=12, MEAL_ABSORPTION_STEPS=24 -> [0.5, 0.5, ..., 0.5] (24x)
        """
        if self.MEAL_ABSORPTION_STEPS <= 0:
            return []
        increment_per_step = total_meal_glucose / self.MEAL_ABSORPTION_STEPS
        return [increment_per_step] * self.MEAL_ABSORPTION_STEPS

    def _create_insulin_absorption_profile(self, bolus_units: float) -> list:
        """Create triangular insulin absorption over INSULIN_ABSORPTION_STEPS.
        
        Peak effect at step INSULIN_ABSORPTION_STEPS // 2 (1 hour of 3-hour window).
        Returns list of insulin units to apply, one per step.
        Example: triangular peak in middle, sum equals bolus_units.
        """
        if self.INSULIN_ABSORPTION_STEPS <= 0:
            return []
        
        # Triangular profile: rises to peak at midpoint, then falls
        n_steps = self.INSULIN_ABSORPTION_STEPS
        peak_idx = n_steps // 2
        profile = []
        
        # Ascending to peak
        for i in range(peak_idx):
            # Linear rise: 0 to 1 as i goes 0 to peak_idx
            profile.append((i + 1) / (peak_idx + 1))
        
        # Descending from peak
        for i in range(peak_idx, n_steps):
            # Linear fall: 1 down to near 0
            profile.append((n_steps - i) / (n_steps - peak_idx))
        
        # Normalize so sum equals bolus_units
        profile_sum = sum(profile)
        if profile_sum > 0:
            profile = [u * bolus_units / profile_sum for u in profile]
        
        return profile

    def _consume_absorption_queue(self, queue: list) -> float:
        """Pop the first element from absorption queue and return it. Returns 0 if empty."""
        if len(queue) > 0:
            return queue.pop(0)
        return 0.0

    def step(self, action: int) -> tuple:
        """Take one step (5 minutes) with realistic absorption dynamics."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Store previous glucose for trend
        self.glucose_prev = self.glucose

        # Track zone transitions BEFORE update
        was_in_hypo = self.glucose < self.glucose_hypo
        was_in_severe_hyper = self.glucose > self.glucose_severe_hyper

        bolus_dose = self.bolus_actions[action]
        self.last_bolus_dose = bolus_dose
        if bolus_dose > 0:
            self.steps_since_last_bolus = 0
        else:
            self.steps_since_last_bolus += 1

        # --- Glucose dynamics with temporal absorption ---
        
        # Basal effect (continuous per step, no queuing)
        basal_effect = -self.basal_rate * (self.dt_minutes / 60.0)
        
        # Insulin sensitivity: peak effect is ~1.5 mg/dL per unit absorbed
        insulin_sensitivity = 1.5
        
        # Bolus: queue the insulin units for gradual absorption
        bolus_effect = 0.0  # Will be updated from absorption queue
        if bolus_dose > 0:
            # Create absorption profile for this bolus
            absorption_profile = self._create_insulin_absorption_profile(bolus_dose)
            self.pending_bolus_units.extend(absorption_profile)
        
        # Consume one step of queued insulin.
        # `insulin_units_this_step` is already a per-step absorbed amount from the IOB profile,
        # so do not apply an additional dt scaling factor here.
        insulin_units_this_step = self._consume_absorption_queue(self.pending_bolus_units)
        bolus_effect = -insulin_units_this_step * insulin_sensitivity
        
        # Meal: queue the meal glucose for gradual absorption
        meal_effect = 0.0  # Will be updated from absorption queue
        if self.stochastic_meals and self.rng.rand() < self.meal_prob_per_step:
            meal_carbs = self.rng.uniform(self.meal_min_carbs, self.meal_max_carbs)
            total_meal_glucose = meal_carbs * 0.5
            # Create absorption profile for this meal
            absorption_profile = self._create_meal_absorption_profile(total_meal_glucose)
            self.pending_meal_glucose.extend(absorption_profile)
        
        # Consume one step of queued meal glucose
        meal_effect = self._consume_absorption_queue(self.pending_meal_glucose)
        
        # Stochastic noise
        noise = self.rng.normal(0, self.glucose_delta_std)

        # Update glucose
        self.glucose = self.glucose + basal_effect + bolus_effect + meal_effect + noise
        self.glucose = np.clip(self.glucose, self.glucose_min, self.glucose_max)

        # --- Reward components ---
        zone_reward = 0.0
        if self.TIR_LOW <= self.glucose <= self.TIR_HIGH:
            zone_reward += self.REWARD_IN_RANGE
        elif self.glucose < self.glucose_hypo:
            zone_reward += self.PENALTY_HYPO
        elif self.glucose < self.TIR_LOW:
            zone_reward += self.PENALTY_LOW
        elif self.glucose > self.glucose_severe_hyper:
            zone_reward += self.PENALTY_SEVERE_HYPER
        else:  # TIR_HIGH < glucose <= severe_hyper
            zone_reward += self.PENALTY_HIGH

        # Configurable insulin penalty for reward sweeps.
        insulin_penalty = -self.insulin_penalty_coeff * bolus_dose
        reward = zone_reward + insulin_penalty

        # --- Event tracking (transitions) ---
        now_in_hypo = self.glucose < self.glucose_hypo
        hypo_event = int(now_in_hypo and not was_in_hypo)

        now_in_severe_hyper = self.glucose > self.glucose_severe_hyper
        severe_hyper_event = int(now_in_severe_hyper and not was_in_severe_hyper)

        # Increment step
        self.step_count += 1
        terminated = self.step_count >= self.episode_length
        truncated = False

        # Build info dict
        info = {
            "glucose": self.glucose,
            "bolus_delivered": bolus_dose,
            "basal_effect": basal_effect,
            "bolus_effect": bolus_effect,
            "meal_effect": meal_effect,
            "noise": noise,
            "zone_reward": zone_reward,
            "insulin_penalty": insulin_penalty,
            "hypo_step": int(now_in_hypo),  # 1 if in hypo zone now
            "hypo_event": hypo_event,  # 1 if just entered hypo zone
            "severe_hyper_step": int(now_in_severe_hyper),  # 1 if in severe hyper zone now
            "severe_hyper_event": severe_hyper_event,  # 1 if just entered severe hyper zone
            "pending_meal_glucose": len(self.pending_meal_glucose),
            "pending_bolus_units": len(self.pending_bolus_units),
        }

        if self.verbose and (self.step_count % 48 == 0 or terminated):
            print(f"[step {self.step_count:3d}] glucose={self.glucose:6.1f} | bolus={bolus_dose:4.1f}U | pending_meal={len(self.pending_meal_glucose)} | pending_bolus={len(self.pending_bolus_units)} | reward={reward:+.2f}")

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info


    def render(self, mode: str = "human") -> None:
        pass

    def close(self) -> None:
        pass


if __name__ == "__main__":
    print("Testing updated GlucoseEnv...")
    try:
        env = GlucoseEnv(verbose=False)
        print(f"✓ Environment created")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: Discrete({len(env.bolus_actions)}) doses: {env.bolus_actions}")
        
        obs, _ = env.reset()
        print(f"✓ Reset: obs shape={obs.shape}")
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        print(f"✓ Info dict keys: {list(info.keys())}")
        print(f"✓ Episode completed {_+1} steps")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
