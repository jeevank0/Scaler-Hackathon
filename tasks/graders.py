from __future__ import annotations

from typing import Dict, List

MIN_REWARD_PER_STEP = -1.75
MAX_REWARD_PER_STEP = 1.40
MAX_CHEMICAL_PER_STEP = 30.0


def clamp_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp_01((value - low) / (high - low))


def grade_yield_performance(total_reward: float, total_steps: int) -> Dict[str, float]:
    min_total = MIN_REWARD_PER_STEP * total_steps
    max_total = MAX_REWARD_PER_STEP * total_steps
    score = normalize(total_reward, min_total, max_total)
    return {"task_id": "task_easy_yield", "score": score}


def grade_chemical_efficiency(
    total_fertilizer: float,
    total_pesticide: float,
    total_steps: int,
) -> Dict[str, float]:
    total_chemical_use = total_fertilizer + total_pesticide
    max_chemical_use = MAX_CHEMICAL_PER_STEP * total_steps
    score = 1.0 - normalize(total_chemical_use, 0.0, max_chemical_use)
    return {"task_id": "task_medium_chemical_efficiency", "score": clamp_01(score)}


def grade_sustainability_balance(
    total_yield: float,
    total_fertilizer: float,
    total_pesticide: float,
) -> Dict[str, float]:
    ratio = total_yield / (total_fertilizer + total_pesticide + 1.0)
    score = ratio / (ratio + 1.0)
    return {"task_id": "task_hard_sustainability_balance", "score": clamp_01(score)}


def grade_all(
    total_reward: float,
    total_yield: float,
    total_fertilizer: float,
    total_pesticide: float,
    total_steps: int,
) -> List[Dict[str, float]]:
    return [
        grade_yield_performance(
            total_reward=total_reward, total_steps=total_steps),
        grade_chemical_efficiency(
            total_fertilizer=total_fertilizer,
            total_pesticide=total_pesticide,
            total_steps=total_steps,
        ),
        grade_sustainability_balance(
            total_yield=total_yield,
            total_fertilizer=total_fertilizer,
            total_pesticide=total_pesticide,
        ),
    ]
