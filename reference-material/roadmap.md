# FarmRL Round-1 Fast Development Roadmap

## Reference Materials

### Introduction

FarmRL is a reinforcement learning project that trains an agent to manage crop farming decisions. Given observable farm conditions such as soil properties, weather, and crop type, the agent learns to control irrigation, fertilizer application, and pesticide use in order to maximise crop yield while maintaining a healthy sustainability score.

The project is grounded in a tabular agricultural dataset and draws conceptual inspiration from the FarmGym simulation framework. Two training paradigms are supported: a classic RL agent via a custom OpenEnv environment, and an optional text-framing path using TRL for language-model-based decision making.

The raw CSV dataset is preprocessed once. The preprocessing adds the Water\_mm column (drawn uniformly from [20, min(Rainfall\_mm, 200)]) and subtracts that value from Rainfall\_mm to preserve water-balance invariance. A lightweight regression model (XGBoost) is then trained on the processed data to serve as the environment's transition model.

---

## Dataset preprocessing requirement

Add a preprocessing script that creates a new variable Water\_mm such that:

Rainfall\_original = Rainfall\_new + Water\_mm

This prevents bias by conserving total water availability.

Script file:

scripts/add\_water\_variable.py

```
"""
add_water_variable.py

Adds a Water_mm column to the farm dataset.
Water is drawn uniformly from [WATER_MIN, Rainfall_mm].
Rainfall_mm is reduced by the water drawn to prevent bias.
"""

import pandas as pd
import numpy as np
import sys

WATER_MIN = 20   # minimum meaningful irrigation (mm)
WATER_MAX = 200  # hard ceiling - avoids flooding; also capped at rainfall

def add_water(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Upper bound: rainfall itself, capped at WATER_MAX
    upper = df["Rainfall_mm"].clip(upper=WATER_MAX)

    # Where rainfall < WATER_MIN we can't irrigate meaningfully — set 0
    can_irrigate = upper >= WATER_MIN
    water = np.where(
        can_irrigate,
        rng.uniform(WATER_MIN, upper.where(can_irrigate, WATER_MIN)),
        0.0
    )

    df["Water_mm"] = np.round(water, 2)
    df["Rainfall_mm"] = np.round(df["Rainfall_mm"] - df["Water_mm"], 2)
    return df


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "farm_data.csv"
    out  = sys.argv[2] if len(sys.argv) > 2 else path.replace(".csv", "_watered.csv")

    df = pd.read_csv(path)
    required = {"Rainfall_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_out = add_water(df)

    print(f"Water_mm  — min: {df_out['Water_mm'].min():.1f}  "
          f"max: {df_out['Water_mm'].max():.1f}  "
          f"mean: {df_out['Water_mm'].mean():.1f}")
    print(f"Rainfall_mm after subtraction — min: {df_out['Rainfall_mm'].min():.1f}  "
          f"mean: {df_out['Rainfall_mm'].mean():.1f}")

    df_out.to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

```

Purpose:

• introduces irrigation variable • prevents data leakage • preserves statistical consistency • improves realism of agent decisions

---

# 3-Phase Fast Development Plan (3–4 hours)

Goal: produce validator-compliant submission with improved reward design.

Scope limitations:

• simple environment dynamics • minimal dataset preprocessing • basic transition model • improved reward shaping only

---

# Phase 1 — OpenEnv Environment (Core functionality)

**Goal:** produce a valid OpenEnv-compliant environment that passes schema and endpoint checks.

Estimated time: **1.5 hours**

---

## Tasks

### 1. Define typed state model (Pydantic)

Keep small but realistic.

Example variables:

```
soil_moisture : float
soil_ph       : float
temperature   : float
rainfall      : float
crop_stage    : int
day           : int

```

Requirements satisfied:

- typed models required by OpenEnv spec
- deterministic state structure

---

### 2. Define typed action model

Discrete actions simplify LLM reliability:

```
water       : float   (0–50)
fertilizer  : float   (0–20)
pesticide   : float   (0–10)

```

Keep ranges bounded to stabilize scoring.

---

### 3. Implement environment class

File:

```
env/farm_env.py

```

Must implement:

```
reset()
step(action)
state()

```

---

### 4. Implement improved reward design (only sophistication added)

Reward must reflect:

- yield improvement
- sustainability balance
- penalty for overuse of chemicals

Example reward:

```
yield_score =
    0.4 * soil_moisture
  + 0.3 * temperature_factor
  + 0.3 * rainfall_factor

resource_penalty =
    0.03 * fertilizer^1.2
  + 0.04 * pesticide^1.3

sustainability_bonus =
    0.2 * exp(-fertilizer/20)
  + 0.2 * exp(-pesticide/10)

reward =
    yield_score
  + sustainability_bonus
  - resource_penalty

```

Characteristics:

- diminishing returns on fertilizer
- discourages excessive pesticide
- stable numeric range
- smooth gradients

---

### 5. Episode termination rule

```
max_days = 30

```

Short episodes ensure runtime < 20 min.

---

### 6. Create openenv.yaml

Define:

```
environment metadata
observation schema
action schema
reward schema
task definitions

```

Ensure field names exactly match Pydantic models.

---

### 7. Implement API wrapper (if required by spec)

Expose:

```
POST /reset
POST /step
GET /state

```

Ensure reset returns valid initial state.

Requirement satisfied:

HF Space ping must return 200.

---

# Phase 2 — inference pipeline + tasks + graders

**Goal:** produce valid evaluation run with structured logs and normalized scores.

Estimated time: **1.5 hours**

---

## Tasks

### 1. Create inference.py in root directory

File location:

```
/inference.py

```

Must:

- load environment
- call LLM via OpenAI client
- run episodes
- log structured output
- compute task scores

---

### 2. Implement OpenAI client usage

Must use env variables:

```
API_BASE_URL
MODEL_NAME
HF_TOKEN

```

LLM prompt format:

```
Farm state:
soil moisture: 34
temperature: 26
rainfall: 3
crop stage: 2

Choose action values:
water
fertilizer
pesticide

```

LLM output expected as JSON:

```
{
 "water": 20,
 "fertilizer": 5,
 "pesticide": 1
}

```

Add fallback defaults if parsing fails.

---

### 3. Define 3 tasks

Tasks must produce score ∈ [0,1].

---

#### Task 1 — yield performance

Measures productivity.

```
score =
normalized(total_reward)

```

---

#### Task 2 — chemical efficiency

Penalizes excessive fertilizer/pesticide.

```
score =
1 - normalized(total_chemical_use)

```

---

#### Task 3 — sustainability balance

Encourages moderate actions.

```
score =
yield / (fertilizer + pesticide + 1)
normalized to 0–1

```

---

### 4. Implement graders

Each grader returns:

```
{
 "task_id": "...",
 "score": float
}

```

Ensure:

```
0 ≤ score ≤ 1

```

Validator requirement.

---

### 5. Implement structured logs

Strict format:

```
[START]
model: MODEL_NAME

[STEP]
step: 1
action: {...}
reward: ...

[STEP]
step: 2
...

[END]
task_scores:
task1: 0.63
task2: 0.71
task3: 0.59

```

Formatting must match specification exactly.

---

### 6. Runtime optimization

Keep small:

```
episodes = 3
steps per episode = 20–30

```

Ensures runtime well below 20 minutes.

---

# Phase 3 — packaging, docker, validation

**Goal:** ensure infrastructure compatibility and reproducibility.

Estimated time: **1 hour**

---

## Tasks

### 1. requirements.txt

Minimal dependencies:

```
pydantic
numpy
pyyaml
openai
fastapi (optional)
uvicorn (optional)

```

Avoid heavy ML libraries.

---

### 2. Dockerfile

Must build automatically.

Example flow:

```
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "inference.py"]

```

Validator requirement satisfied.

---

### 3. environment variables support

Ensure inference.py reads:

```
API_BASE_URL
MODEL_NAME
HF_TOKEN

```

No hardcoding.

---

### 4. basic local tests

Run:

```
python inference.py

```

Verify:

- no crashes
- scores generated
- logs formatted correctly

---

### 5. validation checklist

Confirm:

HF Space can call:

```
reset()
step()
state()

```

Ensure:

- numeric reward returned
- valid JSON outputs
- docker build successful

---

# Final deliverable structure

```
project/
│
├── openenv.yaml
├── inference.py
├── Dockerfile
├── requirements.txt
│
├── env/
│   └── farm_env.py
│
└── tasks/
    └── graders.py

```

---

# Expected outcome

Submission will pass:

- OpenEnv compliance
- structured logging requirement
- 3 task requirement
- reproducibility requirement
- runtime constraint
- docker build requirement
- HF space endpoint validation

---