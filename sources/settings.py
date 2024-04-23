from urllib.parse import urljoin

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_token: str
    api_url: str = Field(default="http://127.0.0.1:8000")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        frozen=True,
        validate_default=True,
    )

    def get_url(self, route: str) -> str:
        return urljoin(self.api_url, route)
