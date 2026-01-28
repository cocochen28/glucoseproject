## Time & Resolution
- Timestep: 5 minutes  
- Episode length: 288 steps (24 hours)  

## Glucose (state dynamics)
- Observation bounds: 40–400 mg/dL  
  - Empirical minimum and maximum from CGM readings  
- Short-term variability (per 5 minutes):  
  - Std of glucose deltas: 6.97 mg/dL  
  - 95th percentile of |Δglucose|: 12.0 mg/dL  
- Typical operating range (dense region):  
  - IQR: 115.0–202.0 mg/dL  
- Rare tail thresholds:  
  - Hypoglycemia: 66.0 mg/dL (2.5th percentile)  
  - Severe hyperglycemia: 292.0 mg/dL (97.5th percentile)  
- Clinical reference (not used for grounding):  
  - 80–180 mg/dL retained for reward reporting and evaluation  

## Meals (exogenous disturbance)
- Typical meal size:  
  - Median: 36.0 g carbohydrates  
- Common meal range:  
  - IQR: 20.0–52.0 g carbohydrates  
- Upper cap for simulation:  
  - 95th percentile: 129.5 g  
- Meal frequency:  
  - Average: 3.81 meals/day  
- Modeling role:  
  - Meals introduce stochastic upward glucose disturbances independent of agent actions  

## Bolus insulin (agent actions)
- Typical bolus dose:  
  - Median: 4.8 units  
- Common dosing range:  
  - IQR: 2.8–7.6 units  
- Upper bound for simulator:  
  - 95th percentile: 12.5 units  
- Action space:  
  - Discrete insulin doses scaled to typical observed values  
- Modeling role:  
  - Primary control action selected by the RL agent  

## Basal insulin (background effect)
- Mean basal rate:  
  - 0.98 units/hour  
- Typical basal range:  
  - IQR: 0.75–1.20 units/hour  
- Variability:  
  - Coefficient of variation (CV): 0.264  
- Modeling choice:  
  - Basal insulin modeled as a constant background glucose-lowering drift  
  - Not treated as a control action due to relatively low variability  
