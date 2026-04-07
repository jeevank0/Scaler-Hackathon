from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.farm_env import FarmAction, FarmEnv, FarmState, FarmStepResult

app = FastAPI(title="FarmRL OpenEnv API", version="0.1.0")
DATASET_PATH = Path(__file__).resolve(
).parents[1] / "farmer_advisor_dataset.csv"
env = FarmEnv(dataset_path=DATASET_PATH)


class ResetRequest(BaseModel):
    seed: int | None = None


@app.post("/reset", response_model=FarmState)
def reset(payload: ResetRequest | None = None) -> FarmState:
    seed = payload.seed if payload is not None else None
    return env.reset(seed=seed)


@app.post("/step", response_model=FarmStepResult)
def step(action: FarmAction) -> FarmStepResult:
    try:
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=FarmState)
def state() -> FarmState:
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
