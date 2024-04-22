from pydantic import BaseModel


class GenerationSchema(BaseModel):
    message: str


class PredictSchema(BaseModel):
    message: str


class PredictionSchema(BaseModel):
    emotion: str
