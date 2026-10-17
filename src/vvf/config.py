"""Configuração central lida do ambiente / .env. Nenhum segredo tem valor padrão."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    fal_key: SecretStr | None = None
    replicate_api_token: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None

    vvf_max_spend_usd: float = Field(default=5.0, ge=0)
    vvf_allow_paid_calls: bool = False
    vvf_projects_dir: Path = Path("projects")
    vvf_cache_dir: Path = Path(".cache/vvf")

    vvf_whisper_model: str = "small"
    vvf_whisper_device: str = "cpu"
    vvf_whisper_compute_type: str = "int8"

    def configured_providers(self) -> dict[str, bool]:
        """Quais integrações têm credencial definida (sem expor valores)."""
        return {
            "anthropic": self.anthropic_api_key is not None,
            "fal": self.fal_key is not None,
            "replicate": self.replicate_api_token is not None,
            "elevenlabs": self.elevenlabs_api_key is not None,
            "llm_router": self.llm_base_url is not None,
        }


def get_settings() -> Settings:
    return Settings()
