"""Application settings using Pydantic Settings with .env support."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderType(StrEnum):
    CLAUDE = "claude"
    OPENAI = "openai"


class MemoryType(StrEnum):
    CONVERSATION = "conversation"
    SUMMARY = "summary"
    VECTOR = "vector"


class Settings(BaseSettings):
    """Central configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), description="Anthropic API key")
    openai_api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")

    # LLM Provider
    default_llm_provider: LLMProviderType = Field(
        default=LLMProviderType.CLAUDE,
        description="Default LLM provider (claude or openai)",
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model identifier",
    )
    max_tokens: int = Field(default=4096, description="Max tokens per LLM response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Default temperature")

    # Agent
    max_agent_steps: int = Field(default=10, ge=1, le=100, description="Max reasoning steps")

    # Sandbox
    sandbox_timeout: int = Field(default=30, ge=1, le=300, description="Code execution timeout (s)")
    sandbox_max_memory_mb: int = Field(default=256, description="Max memory for sandboxed code (MB)")

    # Memory
    memory_type: MemoryType = Field(
        default=MemoryType.CONVERSATION,
        description="Memory backend type",
    )
    chroma_persist_dir: str | None = Field(
        default=None,
        description="ChromaDB persistence directory for vector memory",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_json: bool = Field(default=False, description="Output logs as JSON")

    # API Server
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")

    def get_api_key(self, provider: LLMProviderType | None = None) -> str:
        """Return the API key for the given or default provider."""
        provider = provider or self.default_llm_provider
        if provider == LLMProviderType.CLAUDE:
            return self.anthropic_api_key.get_secret_value()
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton settings instance."""
    return Settings()
