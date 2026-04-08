from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.farm_env import FarmAction, FarmEnv, FarmState, FarmStepResult

app = FastAPI(title="FarmRL OpenEnv API", version="0.1.0")
DATASET_PATH = Path(__file__).resolve(
).parents[1] / "farmer_advisor_dataset.csv"
env = FarmEnv(dataset_path=DATASET_PATH)


class ResetRequest(BaseModel):
    seed: int | None = None


class MCPRequest(BaseModel):
    jsonrpc: str | None = "2.0"
    id: int | str | None = None
    method: str | None = None
    params: dict[str, Any] | None = None


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, str]:
    return {
        "name": "farmrl-phase1",
        "description": "Minimal FarmRL OpenEnv implementation for Round-1 Phase-1.",
    }


@app.get("/schema")
def schema() -> dict[str, dict[str, Any]]:
    state_schema = FarmState.model_json_schema()
    return {
        "action": FarmAction.model_json_schema(),
        "observation": state_schema,
        "state": state_schema,
    }


@app.post("/mcp")
def mcp(payload: MCPRequest) -> dict[str, Any]:
    method = payload.method or ""
    request_id = payload.id

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "farmrl-openenv", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": []},
        }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"ok": True},
    }