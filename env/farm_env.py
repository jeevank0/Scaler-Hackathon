from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class FarmState(BaseModel):
    soil_moisture: float = Field(ge=0.0, le=100.0)
    soil_ph: float = Field(ge=4.0, le=9.0)
    temperature: float
    rainfall: float = Field(ge=0.0)
    crop_stage: int = Field(ge=0)
    day: int = Field(ge=0)


class FarmAction(BaseModel):
    water: float = Field(ge=0.0, le=50.0)
    fertilizer: float = Field(ge=0.0, le=20.0)
    pesticide: float = Field(ge=0.0, le=10.0)


class FarmStepResult(BaseModel):
    observation: FarmState
    reward: float
    done: bool
    info: dict[str, Any]


class FarmEnv:
    """Minimal deterministic OpenEnv-style farm environment for Phase-1."""

    REQUIRED_COLUMNS = {
        "Soil_pH",
        "Soil_Moisture",
        "Temperature_C",
        "Rainfall_mm",
    }

    def __init__(
        self,
        dataset_path: str | Path = "farmer_advisor_dataset.csv",
        seed: int = 42,
        max_days: int = 30,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.max_days = max_days
        self._rng = np.random.default_rng(seed)
        self._dataset = self._load_dataset(self.dataset_path)
        self._row_index = 0
        self._state: FarmState | None = None

    def _load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}")
        return df.reset_index(drop=True)

    def _next_weather_row(self) -> pd.Series:
        self._row_index = (self._row_index + 1) % len(self._dataset)
        return self._dataset.iloc[self._row_index]

    def reset(self, seed: int | None = None) -> FarmState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._row_index = int(self._rng.integers(0, len(self._dataset)))
        row = self._dataset.iloc[self._row_index]

        self._state = FarmState(
            soil_moisture=float(np.clip(row["Soil_Moisture"], 0.0, 100.0)),
            soil_ph=float(np.clip(row["Soil_pH"], 4.5, 8.5)),
            temperature=float(row["Temperature_C"]),
            rainfall=float(np.clip(row["Rainfall_mm"], 0.0, 200.0)),
            crop_stage=0,
            day=0,
        )
        return self._state

    def state(self) -> FarmState:
        if self._state is None:
            raise RuntimeError(
                "Environment is not initialized. Call reset() first.")
        return self._state

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _clamp_score(value: float) -> float:
        return float(max(0.001, min(0.994, float(value))))

    @staticmethod
    def _compute_reward(state: FarmState, action: FarmAction, day: int) -> tuple[float, dict[str, float]]:
        moisture_score = FarmEnv._clamp_score(
            np.clip(state.soil_moisture / 100.0, 0.0, 1.0)
        )
        temperature_factor = FarmEnv._clamp_score(
            np.clip(1.0 - abs(state.temperature - 26.0) / 16.0, 0.0, 1.0)
        )
        rainfall_factor = FarmEnv._clamp_score(
            np.clip(1.0 - abs(state.rainfall - 60.0) / 60.0, 0.0, 1.0)
        )

        yield_score = FarmEnv._clamp_score((
            0.4 * float(moisture_score)
            + 0.3 * float(temperature_factor)
            + 0.3 * float(rainfall_factor)
        ))

        resource_penalty = 0.03 * \
            (action.fertilizer**1.2) + 0.04 * (action.pesticide**1.3)
        sustainability_bonus = 0.2 * np.exp(-action.fertilizer / 20.0) + 0.2 * np.exp(
            -action.pesticide / 10.0
        )

        overuse_penalty = 0.0
        if action.fertilizer > 12.0:
            overuse_penalty += 0.02 * (action.fertilizer - 12.0)
        if action.pesticide > 6.0:
            overuse_penalty += 0.03 * (action.pesticide - 6.0)

        loop_penalty = 0.0
        if day > 20 and action.water == 0.0 and action.fertilizer == 0.0 and action.pesticide == 0.0:
            loop_penalty = 0.1

        reward = float(yield_score + sustainability_bonus -
                       resource_penalty - overuse_penalty - loop_penalty)
        info = {
            "yield_score": float(yield_score),
            "resource_penalty": float(resource_penalty),
            "sustainability_bonus": float(sustainability_bonus),
            "overuse_penalty": float(overuse_penalty),
            "loop_penalty": float(loop_penalty),
        }
        return reward, info

    def step(self, action: FarmAction | dict[str, float]) -> FarmStepResult:
        if self._state is None:
            raise RuntimeError(
                "Environment is not initialized. Call reset() first.")

        action_model = action if isinstance(
            action, FarmAction) else FarmAction(**action)
        previous_state = self._state
        weather = self._next_weather_row()

        day = previous_state.day + 1
        crop_stage = min(5, day // 6)

        temperature = 0.7 * previous_state.temperature + \
            0.3 * float(weather["Temperature_C"])
        rainfall = 0.5 * previous_state.rainfall + \
            0.5 * float(weather["Rainfall_mm"])
        rainfall = self._clip(rainfall, 0.0, 200.0)

        evaporation = max(temperature - 20.0, 0.0) * 0.35
        moisture_gain = 0.12 * rainfall + 0.65 * action_model.water
        moisture_loss = evaporation + 0.5 * crop_stage
        soil_moisture = self._clip(
            previous_state.soil_moisture + moisture_gain - moisture_loss,
            0.0,
            100.0,
        )

        soil_ph = self._clip(
            previous_state.soil_ph - 0.012 *
            action_model.fertilizer + 0.002 * action_model.water,
            4.5,
            8.5,
        )

        self._state = FarmState(
            soil_moisture=soil_moisture,
            soil_ph=soil_ph,
            temperature=float(temperature),
            rainfall=rainfall,
            crop_stage=crop_stage,
            day=day,
        )

        reward, reward_info = self._compute_reward(
            self._state, action_model, day=day)
        done = day >= self.max_days
        return FarmStepResult(
            observation=self._state,
            reward=reward,
            done=done,
            info=reward_info,
        )
