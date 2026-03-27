"""Evaluate simple baseline insulin policies in GlucoseEnv.

Baselines:
1. no_insulin: Always no bolus (action 0)
2. single_threshold: Bolus if glucose > IQR high bound
3. two_threshold: Two-level strategy with high and severe thresholds

Metrics per episode: mean glucose, time-in-range %, hypo/severe hyper counts,
total insulin, total reward, episode length.

Saves detailed results to CSV and prints aggregate summary.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add parent directory to path so we can import from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from glucose_env import GlucoseEnv
from simulator_params import load_params, get_glucose_params


# Standard time-in-range threshold for T1D (80–180 mg/dL)
TIME_IN_RANGE_LOW = 80.0
TIME_IN_RANGE_HIGH = 180.0


class BaselinePolicy:
    """Base class for insulin policies."""

    def __init__(self, env: GlucoseEnv, name: str):
        """Initialize policy with environment reference.

        Args:
            env: GlucoseEnv instance.
            name: Human-readable policy name.
        """
        self.env = env
        self.name = name

    def get_action(self, glucose: float) -> int:
        """Decide bolus action based on current glucose.

        Args:
            glucose: Current glucose level (mg/dL).

        Returns:
            Action index (0–4).
        """
        raise NotImplementedError


class NoInsulinPolicy(BaselinePolicy):
    """Never administer insulin."""

    def __init__(self, env: GlucoseEnv):
        super().__init__(env, "no_insulin")

    def get_action(self, glucose: float) -> int:
        """Always return action 0 (no bolus)."""
        return 0


class SingleThresholdPolicy(BaselinePolicy):
    """Bolus if glucose exceeds IQR high bound."""

    def __init__(self, env: GlucoseEnv, params: Dict):
        super().__init__(env, "single_threshold")
        glucose_cfg = get_glucose_params(params)
        self.high_threshold = glucose_cfg["safe_range_iqr"]["high"]  # IQR high
        self.action_high = 3  # Action 3: high dose (IQR high of boluses)

    def get_action(self, glucose: float) -> int:
        """Return high bolus if glucose > threshold, else no bolus."""
        if glucose > self.high_threshold:
            return self.action_high
        return 0


class TwoThresholdPolicy(BaselinePolicy):
    """Two-level strategy: correction for severe high, standard dose for high."""

    def __init__(self, env: GlucoseEnv, params: Dict):
        super().__init__(env, "two_threshold")
        glucose_cfg = get_glucose_params(params)
        self.severe_high_threshold = glucose_cfg["tail_thresholds"]["severe_hyperglycemia"]  # 97.5 pct
        self.high_threshold = glucose_cfg["safe_range_iqr"]["high"]  # IQR high
        self.action_severe = 4  # Action 4: correction dose (p95)
        self.action_high = 2  # Action 2: median dose

    def get_action(self, glucose: float) -> int:
        """Return action based on two thresholds."""
        if glucose > self.severe_high_threshold:
            return self.action_severe
        elif glucose > self.high_threshold:
            return self.action_high
        return 0


def evaluate_baseline(
    policy: BaselinePolicy,
    env: GlucoseEnv,
    n_episodes: int,
    seed: int,
    params: Dict,
) -> List[Dict]:
    """Run baseline policy over multiple episodes and collect metrics.

    Args:
        policy: Baseline policy instance.
        env: GlucoseEnv instance.
        n_episodes: Number of episodes to run.
        seed: Random seed for reproducibility.
        params: Simulator parameters dict (for thresholds).

    Returns:
        List of dicts, one per episode, with metrics.
    """
    glucose_cfg = get_glucose_params(params)
    hypo_threshold = glucose_cfg["tail_thresholds"]["hypoglycemia"]
    severe_hyper_threshold = glucose_cfg["tail_thresholds"]["severe_hyperglycemia"]

    episodes = []
    rng = np.random.RandomState(seed)

    for ep in range(n_episodes):
        ep_seed = seed + ep  # Different seed per episode for variety
        obs, _ = env.reset(seed=ep_seed)
        glucose = obs[0]

        # Per-episode tracking
        glucoses = [glucose]
        total_reward = 0.0
        total_insulin = 0.0
        hypo_count = 0
        severe_hyper_count = 0

        # Run episode
        for step in range(env.episode_length):
            # Get action from policy
            action = policy.get_action(glucose)

            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            glucose = obs[0]

            # Accumulate metrics
            glucoses.append(glucose)
            total_reward += reward
            total_insulin += info["bolus_delivered"]

            # Count events
            if glucose < hypo_threshold:
                hypo_count += 1
            if glucose > severe_hyper_threshold:
                severe_hyper_count += 1

            if terminated:
                break

        # Compute episode-level metrics
        glucoses = np.array(glucoses)
        mean_glucose = float(np.mean(glucoses))
        time_in_range = float(
            100.0
            * np.sum(
                (glucoses >= TIME_IN_RANGE_LOW) & (glucoses <= TIME_IN_RANGE_HIGH)
            )
            / len(glucoses)
        )

        episode_length = len(glucoses) - 1  # steps taken (not observations)

        episodes.append(
            {
                "baseline": policy.name,
                "episode": ep,
                "mean_glucose": mean_glucose,
                "time_in_range_percent": time_in_range,
                "hypo_count": hypo_count,
                "severe_hyper_count": severe_hyper_count,
                "total_insulin_units": total_insulin,
                "total_reward": total_reward,
                "episode_length": episode_length,
                "seed": ep_seed,
            }
        )

    return episodes


def print_summary(results: List[Dict]) -> None:
    """Print summary table of baseline results.

    Args:
        results: Flattened list of episode metrics dicts.
    """
    # Group by baseline
    by_baseline = {}
    for row in results:
        baseline = row["baseline"]
        if baseline not in by_baseline:
            by_baseline[baseline] = []
        by_baseline[baseline].append(row)

    # Print header
    print("\n" + "=" * 100)
    print("BASELINE EVALUATION SUMMARY (N_EPISODES = {} per baseline)".format(len(by_baseline[list(by_baseline.keys())[0]])))
    print("=" * 100)
    print(
        f"{'Baseline':<20} {'Mean Glucose':<18} {'Time-in-Range %':<18} "
        f"{'Hypo Count':<15} {'Severe Hyper':<15} {'Total Insulin':<15} {'Total Reward':<15}"
    )
    print("-" * 100)

    # Print per-baseline stats
    for baseline in sorted(by_baseline.keys()):
        episodes = by_baseline[baseline]
        metrics = {
            "mean_glucose": [e["mean_glucose"] for e in episodes],
            "time_in_range_percent": [e["time_in_range_percent"] for e in episodes],
            "hypo_count": [e["hypo_count"] for e in episodes],
            "severe_hyper_count": [e["severe_hyper_count"] for e in episodes],
            "total_insulin_units": [e["total_insulin_units"] for e in episodes],
            "total_reward": [e["total_reward"] for e in episodes],
        }

        mean_glucose_mean = np.mean(metrics["mean_glucose"])
        mean_glucose_std = np.std(metrics["mean_glucose"])
        tir_mean = np.mean(metrics["time_in_range_percent"])
        tir_std = np.std(metrics["time_in_range_percent"])
        hypo_mean = np.mean(metrics["hypo_count"])
        hypo_std = np.std(metrics["hypo_count"])
        severe_mean = np.mean(metrics["severe_hyper_count"])
        severe_std = np.std(metrics["severe_hyper_count"])
        insulin_mean = np.mean(metrics["total_insulin_units"])
        insulin_std = np.std(metrics["total_insulin_units"])
        reward_mean = np.mean(metrics["total_reward"])
        reward_std = np.std(metrics["total_reward"])

        print(
            f"{baseline:<20} "
            f"{mean_glucose_mean:7.1f}±{mean_glucose_std:5.1f}      "
            f"{tir_mean:6.1f}±{tir_std:5.1f}      "
            f"{hypo_mean:6.1f}±{hypo_std:4.1f}       "
            f"{severe_mean:6.1f}±{severe_std:4.1f}       "
            f"{insulin_mean:6.1f}±{insulin_std:4.1f}      "
            f"{reward_mean:7.1f}±{reward_std:5.1f}"
        )

    print("=" * 100 + "\n")


def main(argv: List[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate baseline insulin policies in GlucoseEnv"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per baseline (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/baselines_metrics.csv",
        help="Output CSV path (default: artifacts/baselines_metrics.csv)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output (print summary only)",
    )
    args = parser.parse_args(argv)

    print(f"Evaluating baselines with {args.episodes} episodes per policy, seed={args.seed}")

    # Load environment and parameters
    try:
        env = GlucoseEnv(random_seed=args.seed)
        params = load_params()
    except Exception as e:
        print(f"✗ Error loading environment/params: {e}")
        return 1

    # Instantiate policies
    policies = [
        NoInsulinPolicy(env),
        SingleThresholdPolicy(env, params),
        TwoThresholdPolicy(env, params),
    ]

    # Evaluate each baseline
    all_results = []
    for policy in policies:
        print(f"\n[Evaluating {policy.name}...]")
        episodes = evaluate_baseline(policy, env, args.episodes, args.seed, params)
        all_results.extend(episodes)
        print(f"  ✓ Completed {len(episodes)} episodes")

    # Print summary
    print_summary(all_results)

    # Save CSV if not disabled
    if not args.no_csv:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="") as f:
            fieldnames = [
                "baseline",
                "episode",
                "mean_glucose",
                "time_in_range_percent",
                "hypo_count",
                "severe_hyper_count",
                "total_insulin_units",
                "total_reward",
                "episode_length",
                "seed",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"✓ Saved metrics to {out_path}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
