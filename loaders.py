import os
from catboost import CatBoostClassifier
from sqlalchemy import create_engine
import pandas as pd
from config import DATABASE_URL
from loguru import logger

def get_model_path() -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/'
    else:
        MODEL_PATH = os.getcwd()
    logger.info(f"Got model path")
    return MODEL_PATH

def load_models(name: str):
    model_path = get_model_path()
    model = CatBoostClassifier() 
    model.load_model(os.path.join(model_path, name))
    logger.info(f"Model {name} loaded")
    return model

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
        break
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features(table_name: str) -> pd.DataFrame:
    logger.info(f"Loading {table_name}")
    query = f"SELECT * FROM {table_name}"  
    return batch_load_sql(query)