# Glucose Simulator: Temporal Dynamics Reference

## Absorption Parameters

### Meal Absorption
```python
MEAL_ABSORPTION_STEPS = 24  # 5-minute steps
Duration = 24 * 5 min = 120 min = 2 hours
Distribution = Uniform (each step: 1/24 of total)
```

**Example:** 40g carb meal (50 mg/dL / 40g = 1.25 mg/dL per gram)
- Total meal glucose effect: 40 * 0.5 = 20 mg/dL
- Per-step effect: 20 / 24 = 0.833 mg/dL per step for 24 steps
- Glucose rises: 0.83 → 1.67 → 2.50 → ... → 20.0 total over 2 hours

### Insulin Absorption (Bolus/Insulin-On-Board)
```python
INSULIN_ABSORPTION_STEPS = 36  # 5-minute steps  
Duration = 36 * 5 min = 180 min = 3 hours
Peak Effect = Step 18 (~90 minutes / 1.5 hours post-bolus)
Distribution = Triangular (ramps up to peak, then down)
```

**Example:** 12.5U bolus (insulin sensitivity = 1.5 mg/dL per unit)
- Peak effect magnitude: ~12.5 * 1.5 = 18.75 mg/dL reduction at peak
- Distribution: Low → Peak (step 18) → Low, summing to same total as instant case
- Glucose lowers gradually over 3 hours with maximum effect ~1.5h post-injection

## Glucose Dynamics Per Step

```
glucose_t+1 = glucose_t 
             + basal_effect        # Continuous: -basal_rate * (dt/60)
             + bolus_effect        # From IOB queue: next pending insulin unit * sensitivity
             + meal_effect         # From meal queue: next pending glucose from meal  
             + noise               # Stochastic: N(0, glucose_delta_std)
             + [clipped to bounds]
```

### Typical Magnitudes Per Step (dt = 5 min)

| Component | Magnitude | Source |
|-----------|-----------|--------|
| Basal | ~-0.08 mg/dL | Continuous ~0.98 U/hr |
| Bolus (peak) | ~-0.015 to -0.020 mg/dL | Triangular distribution, 1.5 sensitivity |
| Meal (peak) | ~0.83 mg/dL (for 40g meal) | Uniform distribution over 24 steps |
| Noise | ~±7 mg/dL (±1σ) | Glucose variability from EDA |
| **Total change (typical)** | **-20 to +20 mg/dL** | Reasonable physiological range |

**Old behavior:** Instant meal could create ±30-80 mg/dL jumps (unrealistic)

## Absorption Curve Shapes

### Meal: Uniform Distribution
```
mg/dL
per
step
│       ┌────────────────────────────────────┐
│       │  0.833 mg/dL for 40g meal        │
│       │  (constant for all 24 steps)      │
│       └────────────────────────────────────┘
└──────────────────────────────────────────────── Time (120 min)
       1  2  3  ... 12 ... 24
       ↑              ↑
       Meal time    Absorption complete
```

### Insulin: Triangular Distribution  
```
Units
per
step
│              ╱╲
│             ╱  ╲
│            ╱    ╲
│           ╱      ╲
│          ╱        ╲
│         ╱          ╲
│        ╱            ╲
└───────────────────────────── Time (180 min)
         0  6  12  18  24  30  36
              ↑       ↑
           Bolus   Peak (1h)  End (3h)
```

## Queue Evolution Example

### Meal Queue After 40g Carb Meal
```
Step 0: [0.833, 0.833, 0.833, ..., 0.833] (24 entries)
Step 1: [0.833, 0.833, ..., 0.833]        (23 entries, popped 1)
Step 2: [0.833, 0.833, ..., 0.833]        (22 entries, popped 1)
...
Step 24: []                                (empty, fully absorbed)
```

### Bolus Queue After 12.5U Bolus
```
Step 0: [0.35U, 0.70U, 1.05U, ..., peak, ..., 0.03U] (36 entries, triangular)
Step 1: [0.70U, 1.05U, ..., peak, ..., 0.03U]         (35 entries)
...
Step 36: []                                             (empty, fully absorbed)
```

## State Tracking

Info dict now includes:
```python
"pending_meal_glucose": int       # Steps remaining in meal queue
"pending_bolus_units": int        # Steps remaining in insulin queue
```

Useful for debugging and visualization:
```
if info['pending_meal_glucose'] > 0:
    print(f"Meal still absorbing for {info['pending_meal_glucose']} more steps")
if info['pending_bolus_units'] > 0:
    print(f"Bolus still active for {info['pending_bolus_units']} more steps")
```

## Validation Targets

After implementing temporal dynamics, verify:

1. ✓ Meal absorption takes 20-26 steps (should see non-zero meal_effect for ~24 steps)
2. ✓ Bolus absorption takes 30-36 steps (should see non-zero bolus_effect for ~36 steps)
3. ✓ Peak bolus effect occurs around step 12-18 (1-1.5 hours)
4. ✓ No single glucose jumps > 30 mg/dL (physiological limit)
5. ✓ Glucose changes are continuous and smooth (no discontiuities)

See `test_absorption_dynamics.py` for automated validation.

## Adjustment Guide

If you need to change absorption horizons (e.g., for faster-acting insulin):

```python
# In GlucoseEnv class definition:
MEAL_ABSORPTION_STEPS = 24  # Increase for slower absorption, decrease for faster
INSULIN_ABSORPTION_STEPS = 36  # Adjust peak time by changing peak_idx in triangular profile

# To shift peak effect time in insulin curve:
peak_idx = n_steps // 2  # Currently 1/2 way through (peak at 1.5h of 3h)
# peak_idx = n_steps // 3  would shift peak to 1h of 3h
# peak_idx = 2*n_steps // 3 would shift peak to 2h of 3h
```

## Backward Compatibility

✓ Observation space unchanged (5-dim state)
✓ Action space unchanged (5 discrete bolus doses)  
✓ Reward function unchanged (zone-based)
✓ Interface unchanged (step/reset/render methods)
✓ Metrics unchanged (TIR, hypo, severe still tracked)
✓ **All existing code continues to work**

Existing trained models will still load and run, but reward curves will differ slightly due to changed glucose dynamics.
