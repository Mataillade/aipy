from contextlib import suppress
from enum import Enum
from functools import cached_property
from pathlib import Path

import joblib
import pandas
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class Emotion(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ModelService:
    def __init__(
        self,
        model_path: Path = Path("model.joblib"),
        training_data_path: Path = Path("data.csv"),
    ):
        self.model_path = model_path
        self.training_data_path = training_data_path

    @cached_property
    def model(self) -> Pipeline:
        with suppress(FileNotFoundError):
            return joblib.load(self.model_path)

        return self.__create_model()

    async def training(self, dataframe: DataFrame):
        self.__train(self.model, dataframe)

    async def predict(self, message: str) -> Emotion:
        result = self.model.predict((message,))[0]
        return Emotion(result)

    def __create_model(self) -> Pipeline:
        dataframe = pandas.read_csv(self.training_data_path)
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("svc", SVC(class_weight='balanced')),
            ]
        )
        self.__train(model, dataframe)
        return model

    def __train(self, model: Pipeline, dataframe: DataFrame):
        x = dataframe[dataframe['airline_sentiment'] != 'neutral']['text']
        y = dataframe[dataframe['airline_sentiment'] != 'neutral']['airline_sentiment']
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )
        model.fit(x_train, y_train)
        joblib.dump(model, self.model_path)
