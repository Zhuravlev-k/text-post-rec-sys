from datetime import datetime
import pandas as pd
import hashlib
from loguru import logger

def merge_features(
            user_id: int,
            time: datetime,
            user_table: pd.DataFrame,
            post_table: pd.DataFrame,
            order: list
            ) -> pd.DataFrame:
    logger.info(f"Starting merging tables.")
    user = user_table[user_table["user_id"] == user_id]
    users = user.loc[user.index.repeat(post_table.shape[0])].reset_index(drop=True)
    time_stamp = pd.DataFrame({"timestamp":[time]})
    time_stamps = pd.DataFrame({
        "month": time_stamp["timestamp"].dt.month, 
        "day": time_stamp["timestamp"].dt.day, 
        "hour": time_stamp["timestamp"].dt.hour, 
        "minute": time_stamp["timestamp"].dt.minute
        })
    time_stamps = time_stamps.loc[time_stamps.index.repeat(post_table.shape[0])].reset_index(drop=True)
    pivot_table = pd.concat([users, time_stamps, post_table], axis=1)
    pivot_table = pivot_table[order]
    logger.info(f"Tables meged.")
    return pivot_table

def get_user_group(id: int, salt: str = "SALT") -> str:
    value_str = str(id) + salt
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"