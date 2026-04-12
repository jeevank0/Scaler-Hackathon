"""Task grading functions for FarmRL OpenEnv submission.

Each grader returns a normalized score in [0, 1].
"""

from __future__ import annotations

from typing import Dict


def _clamp_score(value: float) -> float:
    """Clamp score values to the validator-safe interval."""
    return max(0.001, min(0.994, float(value)))


def grade_yield_performance(
    total_reward: float = 0.0,
    total_steps: int = 0,
) -> Dict[str, float | str]:
    """Grade yield performance using average reward per step.

    Scale factor keeps scores in a useful range while staying normalized.
    """
    avg_reward = total_reward / total_steps if total_steps > 0 else 0.0
    score = _clamp_score(avg_reward / 10.0)
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
    score = _clamp_score(1.0 - _clamp_score(chemical_per_step / 10.0))
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
    score = _clamp_score(ratio / 5.0)
    return {
        "task_id": "task_hard_sustainability_balance",
        "score": score,
    }


def grade_soil_health(
    avg_soil_moisture: float = 0.0,
    avg_soil_ph: float = 6.5,
) -> Dict[str, float | str]:
    """Grade soil health monitoring based on moisture and pH stability.
    
    Optimal range:
    - Soil moisture: 30-70% (too dry < 20%, too wet > 80%)
    - Soil pH: 6.0-7.5 (acidic < 5.5, alkaline > 8.0)
    """
    # Moisture score: peaks at 50%, penalizes extremes
    moisture_ideal = 50.0
    moisture_deviation = abs(avg_soil_moisture - moisture_ideal)
    moisture_score = _clamp_score(1.0 - _clamp_score(moisture_deviation / 50.0))
    
    # pH score: peaks at 6.8 (neutral), acceptable range 6.0-7.5
    ph_ideal = 6.8
    ph_deviation = abs(avg_soil_ph - ph_ideal)
    ph_score = _clamp_score(1.0 - _clamp_score(ph_deviation / 1.5))
    
    # Combined score (equal weight)
    score = _clamp_score((moisture_score + ph_score) / 2.0)
    
    return {
        "task_id": "task_expert_soil_health",
        "score": score,
    }


def grade_all(
    total_reward: float = 0.0,
    total_yield: float = 0.0,
    total_fertilizer: float = 0.0,
    total_pesticide: float = 0.0,
    total_steps: int = 0,
    avg_soil_moisture: float = 50.0,
    avg_soil_ph: float = 6.8,
) -> Dict[str, Dict[str, float | str]]:
    """Return scores for all 4 tasks in one call."""
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
        "task_expert_soil_health": grade_soil_health(
            avg_soil_moisture=avg_soil_moisture,
            avg_soil_ph=avg_soil_ph,
        ),
    }
