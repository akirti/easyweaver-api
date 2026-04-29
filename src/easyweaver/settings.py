"""EasyWeaver settings — Hybrid: ConfigurationLoader JSON + Pydantic BaseSettings."""
import os
from pathlib import Path

from pydantic_settings import BaseSettings

from . import ENVIRONEMNT_VARIABLE_PREFIX, OS_PROPERTY_SEPRATOR

def _load_config() -> dict:
    """Load configuration from JSON files via ConfigurationLoader.

    Uses the 3-file pipeline: localenv-{env}.json + {env}.json + config.json
    plus optional simulator_file for local dev (server.env.{env}.json).
    """
    try:
        from easyweaver.utils.config import ConfigurationLoader

        config_path = os.environ.get(
            "CONFIG_PATH",
            str(Path(__file__).resolve().parent.parent.parent / "config"),
        )
        environment = os.environ.get("EASYWEAVER_ENVIRONMENT", "dev")

        # Use server env file as simulator for local dev (fills in hardcoded values)
        simulator_file = os.environ.get("SIMULATOR_FILE")
        if not simulator_file:
            candidate = Path(config_path) / "server" / f"server.env.{environment}.json"
            if candidate.exists():
                simulator_file = str(candidate)

        loader = ConfigurationLoader(
            config_path=config_path,
            environment=environment,
            simulator_file=simulator_file,
        )
        return loader.configuration
    except Exception:
        return {}


# Load JSON config once at module import
_config = _load_config()


def _get(path: str, default=None):
    """Get a value from the loaded config by dot-path.

    Returns *default* when the resolved value is still an unresolved
    ``{placeholder}`` string so Pydantic never sees raw template tokens.
    """
    import re

    parts = path.split(".")
    val = _config
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return default
        if val is None:
            return default
    # Guard against unresolved {placeholder} strings leaking through
    if isinstance(val, str) and re.search(r'\{[^{}]+\}', val):
        return default
    return val


class Settings(BaseSettings):
    model_config = {"env_prefix": ENVIRONEMNT_VARIABLE_PREFIX+"_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # App
    app_name: str = _get("environment.app_name", "easyweaver")
    debug: bool = _get("environment.debug", False)
    secret_key: str = _get("environment.authentication.jwt_secret_key", "change-me-in-production")

    # MongoDB (from config pipeline)
    mongo_url: str = ""  # Built from config in __init__

    # Redis
    redis_url: str = _get("environment.redis.url", "redis://localhost:6379/0")

    # Celery
    celery_broker_url: str = _get("environment.celery.broker_url", "redis://localhost:6379/1")
    celery_result_backend: str = _get("environment.celery.result_backend", "redis://localhost:6379/1")

    # Auth
    jwt_secret_key: str = _get("environment.authentication.jwt_secret_key", "change-me-jwt-secret")
    jwt_access_token_expire_minutes: int = _get("environment.authentication.jwt_access_token_expiry_minutes", 30)
    jwt_refresh_token_expire_days: int = 7

    # Auth bridge
    admin_jwt_secret_key: str = ""
    admin_jwt_issuer: str = _get("environment.authentication.jwt_issuer", "easylife-auth")
    admin_jwt_audience: str = _get("environment.authentication.jwt_audience", "easylife-api")

    # Encryption
    fernet_key: str = _get("environment.encryption.fernet_key", "change-me-generate-with-cryptography-fernet")

    # Query limits
    max_result_rows: int = _get("globals.databases.default.max_result_rows", 100_000)
    max_export_rows: int = _get("globals.databases.default.max_export_rows", 500_000)
    result_ttl_seconds: int = _get("globals.databases.default.resul_ttl_seconds", 3600)
    schema_cache_ttl_seconds: int = _get("globals.databases.default.schema_cache_ttl_seconds", 900)
    query_timeout_seconds: int = _get("globals.databases.default.query_timeout_seconds", 300)

    # Concurrency
    max_concurrent_queries: int = _get("globals.databases.default.max_concurrent_queries", 10)
    rate_limit_per_minute: int = _get("environment.redis.rate_limit_max_requests", 120)

    # Connection pooling
    db_pool_min_size: int = _get("globals.databases.default.min_pool_size", 1)
    db_pool_max_size: int = _get("globals.databases.default.max_pool_size", 10)
    redis_max_connections: int = 20

    # Process run limits
    max_process_runs_per_config: int = 50

    # Batched fetch
    batch_default_target_seconds: float = 10.0
    batch_min_size: int = 1_000
    batch_max_size: int = 100_000
    batch_intermediate_ttl_seconds: int = 3600

    # GCS Storage (from config pipeline)
    gcs_bucket_name: str = _get("environment.storage.gcs.bucket_name", "")
    gcs_credentials_path: str = _get("environment.storage.gcs.credentials_path", "")
    gcs_credentials_json: str = ""
    file_storage_type: str = _get("environment.storage.type", "redis")

    # CORS
    cors_origins: list[str] = _get("environment.cors.origins", ["http://localhost:5173"])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Build mongo_url from config if not set via env var
        if not self.mongo_url:
            db_info = _get("databases.metadata.db_info", {})
            if isinstance(db_info, dict) and db_info.get("host"):
                scheme = db_info.get("connection_scheme", "mongodb")
                user = db_info.get("username", "")
                pwd = db_info.get("password", "")
                host = db_info.get("host", "localhost:27017")
                if user and pwd:
                    self.mongo_url = f"{scheme}://{user}:{pwd}@{host}"
                else:
                    self.mongo_url = f"{scheme}://{host}"
            else:
                self.mongo_url = os.environ.get("EASYWEAVER_MONGO_URL", "mongodb://localhost:27018")


settings = Settings()
