import numpy as np, pandas as pd
NUM_FEATURES = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
ENGINEERED = ["temp_diff", "power_w", "wear_torque"]
ALL_NUM = NUM_FEATURES + ENGINEERED
CAT = ["Type"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    df["power_w"] = df["torque"] * df["rpm"] * 2 * np.pi / 60
    df["wear_torque"] = df["tool_wear"] * df["torque"]
    return df
