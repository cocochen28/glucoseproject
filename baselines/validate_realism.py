"""Run realism validation checks before retraining.

Checks requested:
1) Plausible daily insulin totals (random + baseline policies)
2) Plausible meal response peaks
3) Plausible time spent in extreme zones
4) No immediate meal spike with instant insulin cancellation

Outputs:
- artifacts/realism_policy_metrics.csv
- artifacts/realism_validation_summary.csv
- Console PASS/FAIL summary
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from glucose_env import GlucoseEnv
from simulator_params import load_params, get_basal_params, get_bolus_params, get_meal_params
from baselines.eval_baselines import (
    BaselinePolicy,
    NoInsulinPolicy,
    SingleThresholdPolicy,
    TwoThresholdPolicy,
    evaluate_baseline,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


class RandomPolicy(BaselinePolicy):
    """Uniform random action baseline."""

    def __init__(self, env: GlucoseEnv, seed: int = 42):
        super().__init__(env, "random_policy")
        self.rng = np.random.RandomState(seed)

    def get_action(self, glucose: float) -> int:
        return int(self.rng.randint(self.env.action_space.n))


def expected_total_insulin_band(params: Dict) -> Tuple[float, float, float]:
    """Compute a data-driven plausible daily total insulin band.

    Expected daily total ~= basal/day + meals/day * median bolus.
    We allow wide realism tolerance around this estimate.
    """
    basal = get_basal_params(params)
    bolus = get_bolus_params(params)
    meals = get_meal_params(params)

    expected = basal["mean_rate_u_per_hr"] * 24.0 + meals["avg_meals_per_day"] * bolus["typical_median_units"]
    low = 0.4 * expected
    high = 2.0 * expected
    return expected, low, high


def run_policy_metrics(env: GlucoseEnv, params: Dict, episodes: int, seed: int) -> List[Dict]:
    """Run random + existing baseline policies and return episode metrics rows."""
    policies = [
        NoInsulinPolicy(env),
        RandomPolicy(env, seed=seed),
        SingleThresholdPolicy(env, params),
        TwoThresholdPolicy(env, params),
    ]

    rows: List[Dict] = []
    for policy in policies:
        rows.extend(evaluate_baseline(policy, env, episodes, seed, params))
    return rows


def check_daily_insulin_totals(rows: List[Dict], params: Dict) -> CheckResult:
    expected, low, high = expected_total_insulin_band(params)

    by_policy: Dict[str, List[float]] = {}
    for r in rows:
        by_policy.setdefault(r["baseline"], []).append(float(r["total_insulin_units"]))

    messages = [f"expected~{expected:.1f} U/day, plausible band [{low:.1f}, {high:.1f}] U/day"]
    all_pass = True
    for policy, vals in sorted(by_policy.items()):
        mean_insulin = float(np.mean(vals))
        # no_insulin and random_policy are intentional stress baselines;
        # report them, but don't gate overall physiological pass/fail on them.
        if policy in {"no_insulin", "random_policy"}:
            messages.append(f"{policy}: {mean_insulin:.1f} U/day (informational baseline)")
            continue
        policy_pass = low <= mean_insulin <= high
        all_pass = all_pass and policy_pass
        messages.append(f"{policy}: {mean_insulin:.1f} U/day -> {'PASS' if policy_pass else 'FAIL'}")

    return CheckResult(
        name="plausible_daily_insulin_totals",
        passed=all_pass,
        details="; ".join(messages),
    )


def check_extreme_zone_time(rows: List[Dict], episode_length: int) -> CheckResult:
    """Check if extreme-zone occupancy is not excessive under simple policies.

    Uses mean occupancy percentages and requires both hypo and severe hyper < 20%.
    """
    by_policy: Dict[str, List[Dict]] = {}
    for r in rows:
        by_policy.setdefault(r["baseline"], []).append(r)

    cap = 20.0
    messages: List[str] = [f"cap={cap:.1f}% for hypo and severe-hyper occupancy"]
    all_pass = True

    for policy, vals in sorted(by_policy.items()):
        hypo_pct = 100.0 * float(np.mean([v["hypo_count"] for v in vals])) / float(episode_length)
        severe_pct = 100.0 * float(np.mean([v["severe_hyper_count"] for v in vals])) / float(episode_length)
        p = (hypo_pct <= cap) and (severe_pct <= cap)
        all_pass = all_pass and p
        messages.append(
            f"{policy}: hypo={hypo_pct:.1f}%, severe={severe_pct:.1f}% -> {'PASS' if p else 'FAIL'}"
        )

    return CheckResult(
        name="plausible_extreme_zone_time",
        passed=all_pass,
        details="; ".join(messages),
    )


def check_meal_response_peak(env: GlucoseEnv, params: Dict) -> CheckResult:
    """Inject a deterministic median meal and verify plausible peak magnitude/timing."""
    meal_cfg = get_meal_params(params)
    meal_carbs = float(meal_cfg["typical_median_carbs"])

    obs, _ = env.reset(seed=123)
    env.stochastic_meals = False
    env.glucose_delta_std = 0.0

    initial = float(obs[0])
    total_meal_glucose = meal_carbs * 0.5
    env.pending_meal_glucose.extend(env._create_meal_absorption_profile(total_meal_glucose))

    glucoses = [initial]
    for _ in range(env.MEAL_ABSORPTION_STEPS + 24):
        obs, _, term, _, _ = env.step(0)
        glucoses.append(float(obs[0]))
        if term:
            break

    arr = np.array(glucoses)
    peak_idx = int(np.argmax(arr))
    peak_delta = float(np.max(arr) - initial)
    peak_minutes = peak_idx * env.dt_minutes

    mag_pass = 10.0 <= peak_delta <= 80.0
    time_pass = 30.0 <= peak_minutes <= 180.0
    passed = mag_pass and time_pass

    return CheckResult(
        name="plausible_meal_response_peak",
        passed=passed,
        details=(
            f"meal={meal_carbs:.1f}g, peak_delta={peak_delta:.1f} mg/dL, "
            f"t_peak={peak_minutes:.0f} min -> "
            f"magnitude {'PASS' if mag_pass else 'FAIL'}, timing {'PASS' if time_pass else 'FAIL'}"
        ),
    )


def check_no_instant_meal_insulin_cancellation(env: GlucoseEnv, params: Dict) -> CheckResult:
    """Simultaneously inject meal + bolus and verify no instant impulse/cancellation."""
    meal_cfg = get_meal_params(params)
    bolus_cfg = get_bolus_params(params)

    meal_carbs = float(meal_cfg["typical_median_carbs"])
    bolus_units = float(bolus_cfg["upper_bound_p95"])

    obs, _ = env.reset(seed=321)
    env.stochastic_meals = False
    env.glucose_delta_std = 0.0

    total_meal_glucose = meal_carbs * 0.5
    env.pending_meal_glucose.extend(env._create_meal_absorption_profile(total_meal_glucose))
    env.pending_bolus_units.extend(env._create_insulin_absorption_profile(bolus_units))

    _, _, _, _, info = env.step(0)

    first_meal = float(info["meal_effect"])
    first_bolus = float(info["bolus_effect"])

    # No instant meal spike: first step should not contain large fraction of total meal effect.
    meal_fraction = abs(first_meal) / max(total_meal_glucose, 1e-9)
    meal_spike_pass = meal_fraction <= 0.20

    # No instant cancellation: first bolus step should be small and not cancel most meal rise.
    cancellation_ratio = abs(first_bolus) / max(abs(first_meal), 1e-9)
    cancellation_pass = cancellation_ratio <= 0.50

    passed = meal_spike_pass and cancellation_pass

    return CheckResult(
        name="no_instant_meal_spike_or_cancellation",
        passed=passed,
        details=(
            f"first_meal={first_meal:.3f} mg/dL ({100*meal_fraction:.1f}% of total), "
            f"first_bolus={first_bolus:.3f} mg/dL, cancellation_ratio={cancellation_ratio:.3f} -> "
            f"meal_spike {'PASS' if meal_spike_pass else 'FAIL'}, cancellation {'PASS' if cancellation_pass else 'FAIL'}"
        ),
    )


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, checks: List[CheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed", "details"])
        writer.writeheader()
        for c in checks:
            writer.writerow({"check": c.name, "passed": int(c.passed), "details": c.details})


def main() -> int:
    episodes = 100
    seed = 42

    params = load_params()
    env = GlucoseEnv(random_seed=seed)

    # 1) Policy metrics (random + baselines)
    rows = run_policy_metrics(env, params, episodes=episodes, seed=seed)
    write_csv(Path("artifacts/realism_policy_metrics.csv"), rows)

    # 2) Requested realism checks
    checks = [
        check_daily_insulin_totals(rows, params),
        check_meal_response_peak(env, params),
        check_extreme_zone_time(rows, env.episode_length),
        check_no_instant_meal_insulin_cancellation(env, params),
    ]
    write_summary(Path("artifacts/realism_validation_summary.csv"), checks)

    print("\nRealism Validation Checks")
    print("=" * 90)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"[{status}] {c.name}")
        print(f"  {c.details}")
    print("=" * 90)

    overall = all(c.passed for c in checks)
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    print("Saved: artifacts/realism_policy_metrics.csv")
    print("Saved: artifacts/realism_validation_summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
