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
    ):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.difficulty = difficulty
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to OpenEnv task definition format."""
        return {
            "id": self.task_id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
        }


# Define the three required tasks matching the graders
TASKS: List[TaskDefinition] = [
    TaskDefinition(
        task_id="task_easy_yield",
        name="Yield Performance",
        description="Maximize crop yield through optimal irrigation and environmental management. "
                    "Graded on average reward per step.",
        difficulty="easy",
    ),
    TaskDefinition(
        task_id="task_medium_chemical_efficiency",
        name="Chemical Efficiency",
        description="Minimize fertilizer and pesticide usage while maintaining acceptable yields. "
                    "Graded on chemical use efficiency.",
        difficulty="medium",
    ),
    TaskDefinition(
        task_id="task_hard_sustainability_balance",
        name="Sustainability Balance",
        description="Achieve top-tier sustainability by maximizing yield-to-chemical-input ratio. "
                    "Graded on sustainability metrics.",
        difficulty="hard",
    ),
]


def get_all_tasks() -> List[Dict[str, str]]:
    """Return all task definitions in OpenEnv format."""
    return [task.to_dict() for task in TASKS]


def get_task_by_id(task_id: str) -> Dict[str, str] | None:
    """Look up a task definition by ID."""
    for task in TASKS:
        if task.task_id == task_id:
            return task.to_dict()
    return None
