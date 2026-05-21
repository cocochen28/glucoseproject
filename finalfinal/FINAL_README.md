# Reinforcement Learning for Simulated Glucose Control

This project explores whether reinforcement learning (RL) agents can learn glucose-control strategies inside a simulated Type 1 diabetes environment.

This repository implements and evaluates a Deep Q-Network (DQN) agent for insulin-bolus recommendations in a data-driven Type 1 diabetes simulator derived from the OhioT1DM dataset. The focus is on simulator realism, reward alignment, and robust training so that learned policies increase Time-in-Range (TIR) inside the frozen simulator. 

The project focused not only on RL performance, but also on simulator realism, reward alignment, stability, and physiological plausibility. Much of the research involved debugging unrealistic dynamics and validating whether the environment behaved coherently before retraining RL agents.

## Dataset

This project uses the OhioT1DM dataset:
https://www.kaggle.com/datasets/ryanmouton/ohiot1dm

Key extracted signals:
- CGM glucose readings
- Meal carbohydrate intake
- Bolus insulin doses
- Basal insulin rates

These statistics were used to ground simulator parameters such as:
- glucose ranges
- meal distributions
- insulin doses
- variability and noise


## Simulator Design

### Time Resolution
- 5-minute timestep
- 288 steps per episode (24 hours)

### Observation Space
The environment observation was upgraded from a single glucose value to a 5-feature vector:
- glucose
- glucose trend
- time of day
- last bolus dose
- steps since last bolus

### Action Space
Discrete bolus actions:
- 0 U
- 2.8 U
- 4.8 U
- 7.6 U
- 12.5 U

### Physiological Dynamics
The simulator models:
- meal absorption over time
- insulin-on-board delayed action
- basal insulin drift
- stochastic glucose noise

Meals and insulin are released gradually using temporal absorption curves rather than one-step impulses.


## RL Agent

The project uses a Deep Q-Network (DQN) implemented in PyTorch.

Architecture:
- Input: 5 features
- Hidden layers: 128 → 128
- Output: 5 Q-values (one per insulin action)

Training features:
- Replay buffer
- Target network
- Epsilon-greedy exploration
- Safety-aware checkpoint selection
- Early stopping


## Key Research Challenges

### 1. Reward Misalignment
The original reward safe zone (115–202 mg/dL) did not match the evaluation metric (80–180 mg/dL TIR), causing misleading optimization behavior.

### 2. Unrealistic Insulin Dynamics
Initial simulator dynamics treated meals and insulin too simplistically.

Early versions applied meal spikes almost instantly while insulin effects were extremely weak due to accidental double time-scaling inside the insulin-on-board logic. This caused:
- threshold policies to barely affect TIR
- unrealistic insulin totals
- meal disturbances to dominate glucose behavior

The simulator was later upgraded with:
- meal absorption over multiple timesteps
- delayed insulin-on-board curves
- corrected insulin scaling
- internal physiological memory buffers

### 3. Training Instability
DQN training showed strong instability over long runs.

Models often reached their best performance during mid-training before deteriorating later, meaning the final checkpoint was frequently worse than earlier checkpoints.

To stabilize evaluation:
- best-checkpoint saving was added
- early stopping was implemented
- checkpoint ranking prioritized:
  - safety feasibility
  - higher eval TIR
  - higher eval reward

This significantly improved reproducibility and prevented reporting degraded late-training policies.

### 4. Simulator Realism
Significant effort was spent validating whether:
- insulin totals were plausible
- meal spikes behaved realistically
- extreme glucose zones occurred at believable frequencies


## Results

After realism fixes and stabilized training:

- Best eval TIR: 72.4%
- Hypoglycemia events: 1.00
- Severe hyper events: 1.60

Key findings:
- Delayed meal/insulin dynamics significantly improved simulator realism
- Reward alignment improved training interpretability
- Best-checkpoint selection was critical due to DQN instability
- Threshold baselines only improved after fixing insulin scaling bugs


## Limitations

Current simplifications include:
- single generic patient dynamics
- simplified meal absorption
- fixed insulin sensitivity
- stochastic meal timing
- no exercise/stress/hormonal modeling