"""Task grading functions for FarmRL OpenEnv submission.

Each grader returns a normalized score in [0, 1].
"""

from __future__ import annotations

from typing import Dict


def _clamp01(value: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def grade_yield_performance(
    total_reward: float = 0.0,
    total_steps: int = 0,
) -> Dict[str, float | str]:
    """Grade yield performance using average reward per step.

    Scale factor keeps scores in a useful range while staying normalized.
    """
    avg_reward = total_reward / total_steps if total_steps > 0 else 0.0
    score = _clamp01(avg_reward / 10.0)
    return {
        "task_id": "task_easy_yield",
        "score": score,
    }


def grade_chemical_efficiency(
    total_fertilizer: float = 0.0,
    total_pesticide: float = 0.0,
    total_steps: int = 0,
) -> Dict[str, float | str]:
    """Grade chemical efficiency by penalizing chemical use per step."""
    steps = total_steps if total_steps > 0 else 1
    chemical_per_step = (total_fertilizer + total_pesticide) / steps
    score = 1.0 - _clamp01(chemical_per_step / 10.0)
    return {
        "task_id": "task_medium_chemical_efficiency",
        "score": score,
    }


def grade_sustainability_balance(
    total_yield: float = 0.0,
    total_fertilizer: float = 0.0,
    total_pesticide: float = 0.0,
) -> Dict[str, float | str]:
    """Grade sustainability using yield-to-chemical-input ratio."""
    ratio = total_yield / (total_fertilizer + total_pesticide + 1.0)
    score = _clamp01(ratio / 5.0)
    return {
        "task_id": "task_hard_sustainability_balance",
        "score": score,
    }


def grade_all(
    total_reward: float = 0.0,
    total_yield: float = 0.0,
    total_fertilizer: float = 0.0,
    total_pesticide: float = 0.0,
    total_steps: int = 0,
) -> Dict[str, Dict[str, float | str]]:
    """Return scores for all tasks in one call."""
    return {
        "task_easy_yield": grade_yield_performance(
            total_reward=total_reward,
            total_steps=total_steps,
        ),
        "task_medium_chemical_efficiency": grade_chemical_efficiency(
            total_fertilizer=total_fertilizer,
            total_pesticide=total_pesticide,
            total_steps=total_steps,
        ),
        "task_hard_sustainability_balance": grade_sustainability_balance(
            total_yield=total_yield,
            total_fertilizer=total_fertilizer,
            total_pesticide=total_pesticide,
        ),
    }
