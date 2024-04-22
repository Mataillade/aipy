from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_token: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        frozen=True,
        validate_default=True,
    )
