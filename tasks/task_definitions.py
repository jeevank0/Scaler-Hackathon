"""
Task definitions for FarmRL OpenEnv submission.

Defines three progressive difficulty levels for farm management optimization.
Each task is associated with a grading function that evaluates performance.
"""

from typing import Dict, List, Any


class TaskDefinition:
    """Represents a task with metadata and grader association."""

    def __init__(
        self,
        task_id: str,
        name: str,
        description: str,
        difficulty: str,
        grader: str,
    ):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.difficulty = difficulty
        self.grader = grader

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenEnv task definition format."""
        grader_name = self.grader.split(
            ":")[-1] if ":" in self.grader else self.grader
        return {
            "id": self.task_id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "grader": self.grader,
            "grader_fn": self.grader,
            "grader_name": grader_name,
            "grader_endpoint": "/grader",
            "grader_enabled": True,
        }


# Define the three required tasks matching the graders
TASKS: List[TaskDefinition] = [
    TaskDefinition(
        task_id="task_easy_yield",
        name="Yield Performance",
        description="Maximize crop yield through optimal irrigation and environmental management. "
                    "Graded on average reward per step.",
        difficulty="easy",
        grader="tasks.graders:grade_yield_performance",
    ),
    TaskDefinition(
        task_id="task_medium_chemical_efficiency",
        name="Chemical Efficiency",
        description="Minimize fertilizer and pesticide usage while maintaining acceptable yields. "
                    "Graded on chemical use efficiency.",
        difficulty="medium",
        grader="tasks.graders:grade_chemical_efficiency",
    ),
    TaskDefinition(
        task_id="task_hard_sustainability_balance",
        name="Sustainability Balance",
        description="Achieve top-tier sustainability by maximizing yield-to-chemical-input ratio. "
                    "Graded on sustainability metrics.",
        difficulty="hard",
        grader="tasks.graders:grade_sustainability_balance",
    ),
    TaskDefinition(
        task_id="task_expert_soil_health",
        name="Soil Health Monitoring",
        description="Maintain optimal soil conditions by managing moisture and pH levels. "
                    "Graded on soil pH stability and moisture retention within ideal ranges.",
        difficulty="expert",
        grader="tasks.graders:grade_soil_health",
    ),
]


def get_all_tasks() -> List[Dict[str, Any]]:
    """Return all task definitions in OpenEnv format."""
    return [task.to_dict() for task in TASKS]


def get_task_by_id(task_id: str) -> Dict[str, Any] | None:
    """Look up a task definition by ID."""
    for task in TASKS:
        if task.task_id == task_id:
            return task.to_dict()
    return None
