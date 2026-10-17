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
    vvf_library_dir: Path = Path("library/audio")

    vvf_language: str = "pt"
    vvf_scene_threshold: float = Field(default=27.0, gt=0)
    vvf_min_duration_s: float = Field(default=1.0, gt=0)
    vvf_max_duration_s: float = Field(default=3 * 3600.0, gt=0)
    vvf_min_height: int = Field(default=240, gt=0)
    vvf_step_retries: int = Field(default=2, ge=0, le=10)

    vvf_piper_voice: str | None = None
    vvf_piper_bin: str = "piper"
    vvf_elevenlabs_voice_id: str | None = None
    vvf_elevenlabs_model: str = "eleven_multilingual_v2"
    vvf_llm_model: str = "claude-opus-5-5"
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
