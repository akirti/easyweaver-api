"""Tests for easyweaver.settings — Settings loading, defaults, env overrides."""
import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    @pytest.fixture
    def settings(self):
        from easyweaver.settings import Settings
        return Settings()

    def test_app_name_has_default(self, settings):
        assert settings.app_name is not None
        assert isinstance(settings.app_name, str)

    def test_debug_is_bool(self, settings):
        assert isinstance(settings.debug, bool)

    def test_secret_key_has_value(self, settings):
        assert isinstance(settings.secret_key, str)
        assert len(settings.secret_key) > 0

    def test_redis_url_has_default(self, settings):
        assert settings.redis_url.startswith("redis://")

    def test_celery_broker_url_has_default(self, settings):
        assert settings.celery_broker_url.startswith("redis://")

    def test_celery_result_backend_has_default(self, settings):
        assert settings.celery_result_backend.startswith("redis://")

    def test_jwt_secret_key_has_value(self, settings):
        assert isinstance(settings.jwt_secret_key, str)

    def test_jwt_access_token_expire_minutes_positive(self, settings):
        assert settings.jwt_access_token_expire_minutes > 0

    def test_jwt_refresh_token_expire_days_positive(self, settings):
        assert settings.jwt_refresh_token_expire_days > 0

    def test_max_result_rows_positive(self, settings):
        assert settings.max_result_rows > 0

    def test_max_export_rows_positive(self, settings):
        assert settings.max_export_rows > 0

    def test_result_ttl_seconds_positive(self, settings):
        assert settings.result_ttl_seconds > 0

    def test_schema_cache_ttl_seconds_positive(self, settings):
        assert settings.schema_cache_ttl_seconds > 0

    def test_query_timeout_seconds_positive(self, settings):
        assert settings.query_timeout_seconds > 0

    def test_max_concurrent_queries_positive(self, settings):
        assert settings.max_concurrent_queries > 0

    def test_rate_limit_per_minute_positive(self, settings):
        assert settings.rate_limit_per_minute > 0

    def test_db_pool_min_and_max_sizes(self, settings):
        assert settings.db_pool_min_size >= 1
        assert settings.db_pool_max_size >= settings.db_pool_min_size

    def test_redis_max_connections(self, settings):
        assert settings.redis_max_connections > 0

    def test_max_process_runs_per_config(self, settings):
        assert settings.max_process_runs_per_config > 0

    def test_batch_default_target_seconds_positive(self, settings):
        assert settings.batch_default_target_seconds > 0

    def test_batch_min_size_less_than_max(self, settings):
        assert settings.batch_min_size < settings.batch_max_size

    def test_batch_intermediate_ttl_seconds_positive(self, settings):
        assert settings.batch_intermediate_ttl_seconds > 0

    def test_file_storage_type_not_empty(self, settings):
        assert isinstance(settings.file_storage_type, str)

    def test_cors_origins_is_list(self, settings):
        assert isinstance(settings.cors_origins, list)

    def test_mongo_url_not_empty(self, settings):
        assert len(settings.mongo_url) > 0
        assert settings.mongo_url.startswith("mongodb")


# ---------------------------------------------------------------------------
# Settings env var overrides (EASYWEAVER_ prefix)
# ---------------------------------------------------------------------------


class TestSettingsEnvOverrides:
    def test_env_prefix_overrides_debug(self):
        with patch.dict(os.environ, {"EASYWEAVER_DEBUG": "true"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert s.debug is True

    def test_env_prefix_overrides_app_name(self):
        with patch.dict(os.environ, {"EASYWEAVER_APP_NAME": "my-test-app"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert s.app_name == "my-test-app"

    def test_env_prefix_overrides_redis_url(self):
        with patch.dict(os.environ, {"EASYWEAVER_REDIS_URL": "redis://custom-host:9999/2"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert s.redis_url == "redis://custom-host:9999/2"

    def test_env_prefix_overrides_max_result_rows(self):
        with patch.dict(os.environ, {"EASYWEAVER_MAX_RESULT_ROWS": "99999"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert s.max_result_rows == 99999

    def test_env_prefix_overrides_mongo_url(self):
        with patch.dict(os.environ, {"EASYWEAVER_MONGO_URL": "mongodb://custom:27099"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert s.mongo_url == "mongodb://custom:27099"

    def test_env_prefix_overrides_jwt_secret_key(self):
        with patch.dict(os.environ, {"EASYWEAVER_JWT_SECRET_KEY": "super-secret-jwt"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert s.jwt_secret_key == "super-secret-jwt"

    def test_env_prefix_overrides_max_concurrent_queries(self):
        with patch.dict(os.environ, {"EASYWEAVER_MAX_CONCURRENT_QUERIES": "25"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert s.max_concurrent_queries == 25

    def test_env_prefix_overrides_fernet_key(self):
        with patch.dict(os.environ, {"EASYWEAVER_FERNET_KEY": "test-fernet-key"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert s.fernet_key == "test-fernet-key"


# ---------------------------------------------------------------------------
# _get helper
# ---------------------------------------------------------------------------


class TestGetHelper:
    def test_get_returns_default_for_missing_path(self):
        from easyweaver.settings import _get
        result = _get("nonexistent.path", default="fallback")
        assert result == "fallback"

    def test_get_returns_none_for_missing_path_no_default(self):
        from easyweaver.settings import _get
        result = _get("some.missing.key")
        assert result is None

    def test_get_returns_default_for_unresolved_placeholder(self):
        from easyweaver import settings as settings_module
        # Temporarily override _config to contain a placeholder
        original_config = settings_module._config
        try:
            settings_module._config = {"level1": {"key": "{UNRESOLVED_PLACEHOLDER}"}}
            from easyweaver.settings import _get
            result = _get("level1.key", default="safe-default")
            assert result == "safe-default"
        finally:
            settings_module._config = original_config

    def test_get_resolves_nested_path(self):
        from easyweaver import settings as settings_module
        original_config = settings_module._config
        try:
            settings_module._config = {"a": {"b": {"c": "deep_value"}}}
            from easyweaver.settings import _get
            result = _get("a.b.c", default="nope")
            assert result == "deep_value"
        finally:
            settings_module._config = original_config

    def test_get_returns_default_when_intermediate_missing(self):
        from easyweaver import settings as settings_module
        original_config = settings_module._config
        try:
            settings_module._config = {"a": {"b": "value"}}
            from easyweaver.settings import _get
            result = _get("a.x.y", default="fallback")
            assert result == "fallback"
        finally:
            settings_module._config = original_config


# ---------------------------------------------------------------------------
# Mongo URL construction
# ---------------------------------------------------------------------------


class TestMongoUrlConstruction:
    def test_mongo_url_from_env_takes_precedence(self):
        with patch.dict(os.environ, {"EASYWEAVER_MONGO_URL": "mongodb://env-host:27099"}):
            from easyweaver.settings import Settings
            s = Settings()
            assert "env-host" in s.mongo_url

    def test_mongo_url_fallback_when_no_config(self):
        """When both env var and config are absent, a fallback URL is used."""
        from easyweaver import settings as settings_module
        original_config = settings_module._config
        try:
            settings_module._config = {}
            # Ensure no env var override
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("EASYWEAVER_MONGO_URL", None)
                from easyweaver.settings import Settings
                s = Settings()
                assert s.mongo_url.startswith("mongodb://")
        finally:
            settings_module._config = original_config


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestSettingsSingleton:
    def test_settings_singleton_is_settings_instance(self):
        from easyweaver.settings import settings, Settings
        assert isinstance(settings, Settings)

    def test_settings_singleton_is_importable(self):
        from easyweaver.settings import settings
        assert settings is not None
