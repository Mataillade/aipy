from pydantic import BaseModel


class PredictSchema(BaseModel):
    message: str


class PredictionSchema(BaseModel):
    emotion: str
