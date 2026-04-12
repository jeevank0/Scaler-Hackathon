from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

from env.farm_env import FarmAction, FarmEnv, FarmState
from tasks.graders import grade_all

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Set it in shell or in {ENV_FILE}."
        )
    return value


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
TASK_NAME = os.getenv("TASK_NAME", "farm-yield-optimization").strip()
BENCHMARK = os.getenv("BENCHMARK", "farmrl").strip()


def resolve_api_key() -> str:
    api_key = os.getenv("API_KEY", "").strip()
    if api_key:
        return api_key
    return os.getenv("OPENAI_API_KEY", "").strip()


API_KEY = resolve_api_key()

PLACEHOLDER_TOKENS = {
    "your_openai_api_key_here",
    "replace_with_openai_api_key",
    "replace-me",
    "replace_me",
}

EPISODES = 3
STEPS_PER_EPISODE = 20
SUCCESS_SCORE_THRESHOLD = 0.10


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_yield_proxy(state: FarmState) -> float:
    moisture_score = clamp(state.soil_moisture / 100.0, 0.0, 1.0)
    temperature_factor = clamp(
        1.0 - abs(state.temperature - 26.0) / 16.0, 0.0, 1.0)
    rainfall_factor = clamp(1.0 - abs(state.rainfall - 60.0) / 60.0, 0.0, 1.0)
    return 0.4 * moisture_score + 0.3 * temperature_factor + 0.3 * rainfall_factor


def build_prompt(state: FarmState, step: int, recent_actions: list[dict[str, float]]) -> str:
    recent_actions_text = "none"
    if recent_actions:
        recent_actions_text = json.dumps(recent_actions[-3:])
    previous_action_text = "none"
    if recent_actions:
        previous_action_text = json.dumps(recent_actions[-1])

    return (
        "Farm state:\n"
        f"step: {step}\n"
        f"soil moisture: {state.soil_moisture:.2f}\n"
        f"temperature: {state.temperature:.2f}\n"
        f"rainfall: {state.rainfall:.2f}\n"
        f"crop stage: {state.crop_stage}\n"
        f"day: {state.day}\n"
        f"previous action: {previous_action_text}\n"
        f"recent actions: {recent_actions_text}\n\n"
        "Choose action values in bounds:\n"
        "water: 0 to 50\n"
        "fertilizer: 0 to 20\n"
        "pesticide: 0 to 10\n\n"
        "Output must be a single valid JSON object with exactly these numeric keys: "
        "water, fertilizer, pesticide.\n"
        "If the previous action is identical, change at least one field by >= 2 unless safety constraints require otherwise."
    )


def build_client() -> Optional[OpenAI]:
    if not API_BASE_URL:
        raise RuntimeError("Missing required environment variable 'API_BASE_URL'.")

    base_lower = API_BASE_URL.lower()
    if "huggingface.co" in base_lower:
        raise RuntimeError(
            "Disallowed API_BASE_URL host 'huggingface.co' for submission."
        )

    api_key = API_KEY
    if not api_key or api_key.lower() in PLACEHOLDER_TOKENS:
        raise RuntimeError(
            "Missing API key. Expected API_KEY (or OPENAI_API_KEY compatibility fallback)."
        )

    base_host = urlparse(API_BASE_URL).netloc or API_BASE_URL
    print(
        f"[INFO] llm_config base_host={base_host} model={MODEL_NAME}",
        flush=True,
    )
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def coerce_action(payload: Dict[str, Any]) -> FarmAction:
    if "water" not in payload or "fertilizer" not in payload or "pesticide" not in payload:
        raise ValueError(
            "Model response must include water, fertilizer, and pesticide.")

    water = float(payload["water"])
    fertilizer = float(payload["fertilizer"])
    pesticide = float(payload["pesticide"])

    normalized = {
        "water": clamp(water, 0.0, 50.0),
        "fertilizer": clamp(fertilizer, 0.0, 20.0),
        "pesticide": clamp(pesticide, 0.0, 10.0),
    }
    return FarmAction(**normalized)


def choose_fallback_action(state: FarmState, recent_actions: list[dict[str, float]]) -> FarmAction:
    # Rule-based action used when LLM is unavailable or returns invalid output.
    target_moisture = 62.0 if state.crop_stage < 3 else 68.0
    moisture_gap = target_moisture - state.soil_moisture
    rain_adjustment = max(0.0, 50.0 - state.rainfall) * 0.1

    water = clamp(12.0 + 0.8 * moisture_gap + rain_adjustment, 0.0, 50.0)
    fertilizer = clamp(
        (6.0 if state.crop_stage < 4 else 4.0)
        - 0.05 * max(0, state.day - 10)
        - 0.1 * max(state.temperature - 32.0, 0.0),
        0.0,
        20.0,
    )

    pesticide = 1.0
    if state.crop_stage >= 2 and state.rainfall > 70.0:
        pesticide = 3.0
    pesticide = clamp(pesticide, 0.0, 10.0)

    action = {
        "water": water,
        "fertilizer": fertilizer,
        "pesticide": pesticide,
    }

    if recent_actions and action == recent_actions[-1]:
        action["water"] = clamp(action["water"] + 2.0, 0.0, 50.0)

    return FarmAction(**action)


def choose_action(
    client: OpenAI,
    state: FarmState,
    step: int,
    recent_actions: list[dict[str, float]],
) -> FarmAction:
    prompt = build_prompt(state, step=step, recent_actions=recent_actions)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a farm operations optimizer. "
                    "Produce state-dependent control decisions and avoid repetitive action loops. "
                    "Return strict JSON with numeric keys: water, fertilizer, pesticide. "
                    "No markdown, no prose, no extra keys. "
                    "Do not output the same action repeatedly across steps unless explicitly necessary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        top_p=0.9,
        frequency_penalty=0.6,
        response_format={"type": "json_object"},
        seed=42 + step,
        max_tokens=160,
    )

    content = (completion.choices[0].message.content or "").strip()
    payload = extract_json_object(content)
    if payload is None:
        raise ValueError("Model did not return valid JSON action payload.")
    return coerce_action(payload)


def to_action_string(action: FarmAction) -> str:
    return json.dumps(action.model_dump(), separators=(",", ":"), sort_keys=True)


def log_start() -> None:
    print(
        f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: FarmAction, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    done_value = str(done).lower()
    action_str = to_action_string(action)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_inference() -> None:
    dataset_path = Path(__file__).resolve().parent / \
        "farmer_advisor_dataset.csv"
    env = FarmEnv(dataset_path=dataset_path, seed=42, max_days=30)
    client = build_client()

    total_reward = 0.0
    total_yield = 0.0
    total_fertilizer = 0.0
    total_pesticide = 0.0
    total_steps = 0
    rewards: list[float] = []
    recent_actions: list[dict[str, float]] = []
    aborted = False
    llm_attempts = 0
    llm_successes = 0
    llm_failures = 0

    log_start()

    for episode in range(EPISODES):
        state = env.reset(seed=42 + episode)

        for _ in range(STEPS_PER_EPISODE):
            step_error: Optional[str] = None

            llm_attempts += 1
            try:
                action = choose_action(
                    client=client,
                    state=state,
                    step=total_steps + 1,
                    recent_actions=recent_actions,
                )
                llm_successes += 1
            except Exception as exc:
                llm_failures += 1
                step_error = f"llm_error:{exc.__class__.__name__}"
                raise RuntimeError(step_error) from exc

            try:
                step_result = env.step(action)
            except Exception as exc:
                aborted = True
                log_step(
                    step=total_steps + 1,
                    action=action,
                    reward=0.0,
                    done=True,
                    error=f"env_error:{exc.__class__.__name__}",
                )
                break

            total_steps += 1
            total_reward += step_result.reward
            total_yield += compute_yield_proxy(step_result.observation)
            total_fertilizer += action.fertilizer
            total_pesticide += action.pesticide
            rewards.append(step_result.reward)
            recent_actions.append(action.model_dump())

            log_step(
                step=total_steps,
                action=action,
                reward=step_result.reward,
                done=step_result.done,
                error=step_error,
            )
            state = step_result.observation

            if step_result.done:
                break

        if aborted:
            break

    if llm_attempts == 0:
        raise RuntimeError("No LLM calls were attempted during inference.")

    print(
        f"[INFO] llm_calls attempts={llm_attempts} successes={llm_successes} failures={llm_failures}",
        flush=True,
    )

    task_scores = grade_all(
        total_reward=total_reward,
        total_yield=total_yield,
        total_fertilizer=total_fertilizer,
        total_pesticide=total_pesticide,
        total_steps=total_steps,
    )
    overall_score = sum(item["score"]
                        for item in task_scores) / len(task_scores)
    overall_score = clamp(overall_score, 0.0, 1.0)
    success = overall_score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=total_steps,
            score=overall_score, rewards=rewards)


def main() -> int:
    try:
        run_inference()
    except Exception as exc:
        fatal_error = re.sub(
            r"\s+",
            " ",
            f"{exc.__class__.__name__}:{exc}",
        ).strip()
        print(f"[FATAL] error={fatal_error}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
