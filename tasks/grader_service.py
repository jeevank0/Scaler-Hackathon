"""
Grader service for evaluating episode results against task-specific criteria.

Provides score normalization, threshold-based pass/fail evaluation, and feedback.
"""

from __future__ import annotations

from typing import Any, Dict

from tasks.graders import (
    grade_yield_performance,
    grade_chemical_efficiency,
    grade_sustainability_balance,
)
from tasks.task_definitions import get_task_by_id


# Difficulty-level specific pass thresholds (score in [0, 1])
PASS_THRESHOLDS = {
    "easy": 0.30,
    "medium": 0.50,
    "hard": 0.70,
}

# Task-to-grader function mapping
GRADER_FUNCTIONS = {
    "task_easy_yield": grade_yield_performance,
    "task_medium_chemical_efficiency": grade_chemical_efficiency,
    "task_hard_sustainability_balance": grade_sustainability_balance,
}


class GraderResult:
    """Result of grading an episode against a task."""
    
    def __init__(
        self,
        task_id: str,
        score: float,
        passed: bool,
        feedback: str,
        difficulty: str,
    ):
        self.task_id = task_id
        self.score = score
        self.passed = passed
        self.feedback = feedback
        self.difficulty = difficulty
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        return {
            "task_id": self.task_id,
            "score": round(self.score, 3),
            "passed": self.passed,
            "feedback": self.feedback,
            "difficulty": self.difficulty,
        }


def evaluate_episode(
    task_id: str,
    total_reward: float = 0.0,
    total_yield: float = 0.0,
    total_fertilizer: float = 0.0,
    total_pesticide: float = 0.0,
    total_steps: int = 0,
) -> GraderResult | None:
    """Evaluate an episode result against a specific task.
    
    Args:
        task_id: Task identifier (must match a registered task)
        total_reward: Cumulative reward across episode
        total_yield: Total yield achieved
        total_fertilizer: Total fertilizer applied
        total_pesticide: Total pesticide applied
        total_steps: Number of steps taken
    
    Returns:
        GraderResult with score, pass status, and feedback, or None if task not found.
    """
    # Validate task exists
    task_def = get_task_by_id(task_id)
    if not task_def:
        return None
    
    difficulty = task_def.get("difficulty", "unknown")
    pass_threshold = PASS_THRESHOLDS.get(difficulty, 0.5)
    
    # Get grader function for this task
    grader_fn = GRADER_FUNCTIONS.get(task_id)
    if not grader_fn:
        return None
    
    # Call grader and extract score
    if task_id == "task_easy_yield":
        grader_result = grader_fn(total_reward=total_reward, total_steps=total_steps)
    elif task_id == "task_medium_chemical_efficiency":
        grader_result = grader_fn(
            total_fertilizer=total_fertilizer,
            total_pesticide=total_pesticide,
            total_steps=total_steps,
        )
    elif task_id == "task_hard_sustainability_balance":
        grader_result = grader_fn(
            total_yield=total_yield,
            total_fertilizer=total_fertilizer,
            total_pesticide=total_pesticide,
        )
    else:
        return None
    
    score = grader_result.get("score", 0.0)
    passed = score >= pass_threshold
    
    # Generate feedback based on score and difficulty
    feedback = _generate_feedback(score, passed, difficulty)
    
    return GraderResult(
        task_id=task_id,
        score=score,
        passed=passed,
        feedback=feedback,
        difficulty=difficulty,
    )


def _generate_feedback(score: float, passed: bool, difficulty: str) -> str:
    """Generate concise feedback based on score and difficulty."""
    threshold = PASS_THRESHOLDS.get(difficulty, 0.5)
    
    if passed:
        if score >= 0.9:
            return f"Excellent performance on {difficulty} task (score: {score:.1%})"
        elif score >= threshold + 0.15:
            return f"Strong performance on {difficulty} task (score: {score:.1%})"
        else:
            return f"Passing {difficulty} task (score: {score:.1%})"
    else:
        gap = threshold - score
        return (
            f"Did not meet {difficulty} threshold ({score:.1%} < {threshold:.1%}, "
            f"gap: {gap:.1%})"
        )
