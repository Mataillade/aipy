from io import BytesIO

import pandas
from fastapi import FastAPI, HTTPException, UploadFile

from sources.model import Emotion, ModelService
from sources.schemas import GenerationSchema, PredictionSchema, PredictSchema

app = FastAPI()
model_service = ModelService()


@app.post("/training", status_code=204)
async def training(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=415)

    buffer = BytesIO(await file.read())
    dataframe = pandas.read_csv(buffer)
    await model_service.training(dataframe)


@app.post("/predict")
async def predict(schema: PredictSchema) -> PredictionSchema:
    emotion = await model_service.predict(schema.message)
    return PredictionSchema(emotion=emotion)


@app.get("/model")
async def model(emotion: Emotion = Emotion.POSITIVE) -> GenerationSchema:
    message = await model_service.generate(emotion)
    return GenerationSchema(message=message)


def main():
    import uvicorn

    uvicorn.run(app, use_colors=True)


if __name__ == "__main__":
    main()
