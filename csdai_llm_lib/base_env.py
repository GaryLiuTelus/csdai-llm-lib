from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrimarySettings(BaseSettings):
    """Runtime configuration sourced from environment variables (or a .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_id: str = Field(default="", min_length=1)
    llm_base_url: str = "https://proxy.fuelix.ai"
    llm_api_key_secret_name: str = Field(default="", min_length=1)
    # Env var is NOTIFICATION_WEBHOOK for backwards compatibility.
    notification_webhook_secret_name: str = Field(
        default="GCHAT_WEBHOOK", validation_alias="NOTIFICATION_WEBHOOK"
    )
    log_level: str = "debug"


@lru_cache
def get_base_settings() -> PrimarySettings:
    return PrimarySettings()
