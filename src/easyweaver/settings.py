from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "EW_", "env_file": ".env", "env_file_encoding": "utf-8"}

    # App
    app_name: str = "easyweaver"
    debug: bool = False
    secret_key: str = "change-me-in-production"

    # Database
    database_url: str = "postgresql+asyncpg://easyweaver:easyweaver@localhost:5432/easyweaver"

    # MongoDB
    mongo_url: str = "mongodb://easyweaver:easyweaver@localhost:27018"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Auth
    jwt_secret_key: str = "change-me-jwt-secret"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # Encryption
    fernet_key: str = "change-me-generate-with-cryptography-fernet"

    # Query limits
    max_result_rows: int = 100_000
    max_export_rows: int = 500_000
    result_ttl_seconds: int = 3600
    schema_cache_ttl_seconds: int = 900
    query_timeout_seconds: int = 300

    # Concurrency
    max_concurrent_queries: int = 10
    rate_limit_per_minute: int = 120

    # Connection pooling
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    redis_max_connections: int = 20

    # Process run limits
    max_process_runs_per_config: int = 50

    # GCS Storage
    gcs_bucket_name: str = ""
    gcs_credentials_path: str = ""
    gcs_credentials_json: str = ""
    file_storage_type: str = "redis"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
