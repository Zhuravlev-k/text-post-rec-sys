from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import List
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from config import DATABASE_URL, order
from loaders import load_models,  load_features
from preprocess import merge_features, get_user_group
from loguru import logger
# формат ответа
class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    
    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]
# подключение к базе
SQLALCHEMY_DATABASE_URL = DATABASE_URL

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
# загрузка моделей
model_control = load_models("model_control")
logger.info(f"Control model loaded")
model_test = load_models("model_test")
logger.info(f"Test model loaded.")

# загрузка предобработанных данных
prepared_users = load_features("kzh_user_data")
logger.info(f"Users table loaded.")
prepared_posts_test = load_features("kzh_post_data_emb")
logger.info(f"Tokenized posts loaded.")
prepared_posts_control = load_features("kzh_post_data_base")
logger.info(f"TF-IDF posts loaded.")
# загрузка необработанных постов для выдачи
post_text_select = "SELECT * FROM public.post_text_df" 
posts = pd.read_sql(post_text_select, engine)
logger.info(f"Posts table loaded")
# запускаем приложение
app = FastAPI()

def db_connect():
    with SessionLocal() as session:
        return session

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
		id: int, 
		time: datetime, 
		limit: int = 10) -> Response:
    user_group = get_user_group(id)
    logger.info(f"User id {id} passed to group: {user_group}")
    # определяем к какой группе A/B теста принадлежит пользователь
    # в зависимости от группы грузим соответсвующие таблицы и модель
    if user_group == "control":
        model = model_control
        pivot_table = merge_features(id, time, prepared_users, prepared_posts_control, order)
    elif user_group == "test":
        model = model_test
        pivot_table = merge_features(id, time, prepared_users, prepared_posts_test, order)
    else:
        raise ValueError("unknown group")
    
    pivot_table.drop(["user_id"], axis=1, inplace=True)
    preds = pd.DataFrame(model.predict_proba(pivot_table), columns=["prob_0", "prob_1"])
    logger.info(f"Made predictions.")
    res_table = posts.join(preds).sort_values(by="prob_1", ascending=False).head(limit)
    response = []
    for i in range(limit):
        response.append(PostGet(id=res_table['post_id'].iloc[i], text=res_table['text'].iloc[i], topic=res_table['topic'].iloc[i]))
    if response == []:
        raise HTTPException(418)
    else:
        return Response(recommendations=response, exp_group=user_group)

if __name__ == "__main__":
    # Простейший тест
    time = datetime(2023, 1, 3, 12, 59)
    print(recommended_posts(id=666, time=time))
    print("seems everything is ok")