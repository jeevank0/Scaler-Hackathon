from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.farm_env import FarmAction, FarmEnv, FarmState, FarmStepResult
from tasks.task_definitions import get_all_tasks
from tasks.grader_service import evaluate_episode

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


class GraderRequest(BaseModel):
    """Episode result to evaluate against a task."""
    task_id: str
    total_reward: float = 0.0
    total_yield: float = 0.0
    total_fertilizer: float = 0.0
    total_pesticide: float = 0.0
    total_steps: int = 0
    avg_soil_moisture: float = 50.0
    avg_soil_ph: float = 6.8


class GraderResponse(BaseModel):
    """Evaluation result with score and pass status."""
    task_id: str
    score: float
    passed: bool
    feedback: str
    difficulty: str


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


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    """Return all available tasks with grader associations.

    OpenEnv requires each submission to declare minimum 3 tasks,
    each associated with a grading function.
    """
    task_items = get_all_tasks()
    grader_map = {
        item["id"]: item.get("grader")
        for item in task_items
        if item.get("id") and item.get("grader")
    }
    return {
        "tasks": task_items,
        "graders": grader_map,
        "tasks_with_graders": len(grader_map),
    }


@app.post("/grader", response_model=GraderResponse)
def grader(payload: GraderRequest) -> GraderResponse:
    """Evaluate an episode result against a specific task.

    Accepts episode metrics, normalizes the reward to a standard scoring scale
    ([0, 1]), evaluates against difficulty-level thresholds, and returns the
    normalized score, pass status, and feedback.

    Args:
        payload: Episode result with task_id and episode metrics

    Returns:
        GraderResponse with score, pass status, and feedback

    Raises:
        HTTPException 400: If task_id is not recognized
    """
    result = evaluate_episode(
        task_id=payload.task_id,
        total_reward=payload.total_reward,
        total_yield=payload.total_yield,
        total_fertilizer=payload.total_fertilizer,
        total_pesticide=payload.total_pesticide,
        total_steps=payload.total_steps,
        avg_soil_moisture=payload.avg_soil_moisture,
        avg_soil_ph=payload.avg_soil_ph,
    )

    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{payload.task_id}' not found or grader unavailable.",
        )

    return GraderResponse(**result.to_dict())


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
