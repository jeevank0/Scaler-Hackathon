"""
add_water_variable.py

Adds a Water_mm column to the farm dataset.
Water is drawn uniformly from [WATER_MIN, Rainfall_mm].
Rainfall_mm is reduced by the water drawn to prevent bias.
"""

import sys

import numpy as np
import pandas as pd

WATER_MIN = 20
WATER_MAX = 200


def add_water(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()

    upper = df["Rainfall_mm"].clip(upper=WATER_MAX)
    can_irrigate = upper >= WATER_MIN
    water = np.where(
        can_irrigate,
        rng.uniform(WATER_MIN, upper.where(can_irrigate, WATER_MIN)),
        0.0,
    )

    df["Water_mm"] = np.round(water, 2)
    df["Rainfall_mm"] = np.round(df["Rainfall_mm"] - df["Water_mm"], 2)
    return df


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "farm_data.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace(
        ".csv", "_watered.csv")

    df = pd.read_csv(path)
    required = {"Rainfall_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_out = add_water(df)

    print(
        f"Water_mm - min: {df_out['Water_mm'].min():.1f} "
        f"max: {df_out['Water_mm'].max():.1f} "
        f"mean: {df_out['Water_mm'].mean():.1f}"
    )
    print(
        "Rainfall_mm after subtraction - "
        f"min: {df_out['Rainfall_mm'].min():.1f} "
        f"mean: {df_out['Rainfall_mm'].mean():.1f}"
    )

    df_out.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
