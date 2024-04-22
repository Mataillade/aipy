from contextlib import suppress
from enum import Enum
from functools import cached_property
from pathlib import Path

import joblib
import pandas
from openai import OpenAI
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sources.settings import Settings


class Emotion(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ModelService:
    def __init__(
        self,
        model_path: Path = Path("model.joblib"),
        training_data_path: Path = Path("data.csv"),
        settings: Settings = Settings(),
    ):
        self.model_path = model_path
        self.training_data_path = training_data_path
        self.settings = settings

    @cached_property
    def model(self) -> Pipeline:
        with suppress(FileNotFoundError):
            return joblib.load(self.model_path)

        return self.__create_model()

    @cached_property
    def client(self) -> OpenAI:
        return OpenAI(api_key=self.settings.openai_token)

    async def training(self, dataframe: DataFrame):
        self.__train(self.model, dataframe)

    async def predict(self, message: str) -> Emotion:
        result = self.model.predict((message,))[0]
        return Emotion(result)

    async def generate(self, emotion: Emotion) -> str:
        prompt = (
            "You are a generator of tweet messages. Your goal is to produce a "
            "tweet refering to a specific topic. The topic is: What was your "
            "experience with your last airline used? Your job is to chose "
            "randomly an airline company between (Virgin America, United, "
            "Southwest, delta, Us Airways, American). Depending on the "
            "sentiments, you have to generte a tweet accordingly to the "
            f"sentiment (Positive or negative). Sentiment : {emotion}"
        )
        completion = self.client.chat.completions.create(
            messages=(
                {
                    "role": "user",
                    "content": prompt,
                },
            ),
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7,
            n=1,
        )
        return completion.choices[0].message.content.strip('"')

    def __create_model(self) -> Pipeline:
        dataframe = pandas.read_csv(self.training_data_path)
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("svc", SVC(class_weight="balanced")),
            ]
        )
        self.__train(model, dataframe)
        return model

    def __train(self, model: Pipeline, dataframe: DataFrame):
        condition = dataframe["airline_sentiment"] != "neutral"
        x = dataframe[condition]["text"]
        y = dataframe[condition]["airline_sentiment"]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )
        model.fit(x_train, y_train)
        joblib.dump(model, self.model_path)
