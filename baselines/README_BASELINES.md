# Baseline Evaluation Script

Evaluates three simple insulin policies on the glucose simulator before RL training.

## Baselines

1. **no_insulin**: Never administer bolus insulin.
2. **single_threshold**: Bolus if glucose > IQR high bound (from EDA).
3. **two_threshold**: Two-level strategy with separate thresholds for high and severe hyperglycemia.

## Thresholds & Actions

All thresholds and action doses are loaded from `simulator_params.py` (derived from EDA):
- **single_threshold**: HIGH_THRESHOLD = IQR high (115–202 mg/dL range, uses high bound), A_HIGH = action 3 (7.6 U)
- **two_threshold**: 
  - SEVERE_HIGH_THRESHOLD = 97.5th percentile (severe hyperglycemia tail threshold)
  - HIGH_THRESHOLD = IQR high
  - A_SEVERE = action 4 (correction, 12.5 U)
  - A_HIGH = action 2 (median, 4.8 U)

## Usage

```bash
python scripts/eval_baselines.py --episodes 20 --seed 42 --out artifacts/baselines_metrics.csv
```

### Arguments
- `--episodes N`: Number of episodes per baseline (default: 20)
- `--seed SEED`: Random seed for reproducibility (default: 42)
- `--out PATH`: Output CSV path (default: `artifacts/baselines_metrics.csv`)
- `--no-csv`: Print summary only, skip CSV output

## Metrics

Per episode:
- **mean_glucose**: Average glucose level over episode
- **time_in_range_percent**: % steps in 80–180 mg/dL (standard T1D range)
- **hypo_count**: Steps below hypoglycemia threshold (2.5th percentile)
- **severe_hyper_count**: Steps above severe hyperglycemia threshold (97.5th percentile)
- **total_insulin_units**: Sum of bolus doses delivered
- **total_reward**: Cumulative environment reward
- **episode_length**: Number of steps (usually 288 = 24 hours)

## Output

**Terminal**: Summary table with mean ± std for each metric per baseline.

**CSV**: One row per episode with all metrics + baseline name + seed.

Example:
```
baseline,episode,mean_glucose,time_in_range_percent,hypo_count,severe_hyper_count,total_insulin_units,total_reward,episode_length,seed
no_insulin,0,218.0,30.4,0,29,0.0,-485.0,288,42
single_threshold,0,205.9,30.4,0,8,1451.6,-439.2,288,42
...
```

## Interpretation

The baseline results provide a **performance floor** for comparison:
- **no_insulin**: Worst case; shows impact of untreated glucose fluctuations
- **single_threshold**: Simple heuristic; often sufficient for moderate high glucose
- **two_threshold**: More aggressive for severe hyperglycemia; better separation of concerns

RL agents should outperform these baselines, especially on:
- Time-in-range %
- Hypo/severe hyper counts
- Total insulin use (with comparable glucose control)
