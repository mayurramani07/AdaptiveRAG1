from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All service connectivity comes from here - never hardcode localhost or
    a Docker service name in client code (FR27, NFR9)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "development"

    opensearch_url: str = ""
    opensearch_user: str = ""
    opensearch_password: str = ""

    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""

    redis_url: str = ""

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    # Separate model for ingestion-time extraction (SS2.3 tier: "Small") -
    # must never be the same model as groq_model, which is reserved for the
    # one expensive generation call (FR18, NFR4). gpt-oss-120b, not a true
    # small model, is the fallback here because Groq currently gates the
    # actually-small chat models (llama-3.1-8b-instant, llama-3.3-70b) behind
    # Enterprise plans - verified against console.groq.com/docs/models on
    # 2026-09-15. Swap this back to a real small model if/when one becomes
    # free-tier accessible again.
    groq_extraction_model: str = "openai/gpt-oss-120b"

    api_key: str = ""
    rate_limit_per_minute: int = 60

    cache_ttl_seconds: int = 86400
    embedding_provider: str = ""


def get_settings() -> Settings:
    return Settings()
