"""Tests for easyweaver.utils.config — ConfigurationLoader and ConfigValueSimulator."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from easyweaver.utils.config import ConfigurationLoader, ConfigValueSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# ConfigurationLoader — legacy (single-file) mode
# ---------------------------------------------------------------------------


class TestLegacyMode:
    def test_loads_config_from_json_file(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        _write_json(cfg_file, {"host": "localhost", "port": 5432})

        loader = ConfigurationLoader(config_path=str(tmp_path))
        assert loader.configuration["host"] == "localhost"
        assert loader.configuration["port"] == 5432

    def test_missing_config_file_results_in_empty_config(self, tmp_path, monkeypatch):
        # Use a subdirectory with no config.json so the fallback doesn't pick
        # up a real config.json from the working directory.
        # Also clear EASYWEAVER_* env vars that might be set by other tests.
        sub = tmp_path / "empty_cfg_dir"
        sub.mkdir()
        for key in list(os.environ):
            if key.startswith("EASYWEAVER"):
                monkeypatch.delenv(key, raising=False)
        loader = ConfigurationLoader(config_path=str(sub))
        assert loader.configuration == {}

    def test_env_var_override_in_legacy_mode(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        _write_json(cfg_file, {"timeout": 30})

        env = {"EASYWEAVER_TIMEOUT": "99"}
        with patch.dict(os.environ, env, clear=False):
            loader = ConfigurationLoader(config_path=str(tmp_path))

        assert loader.configuration["timeout"] == 99

    def test_env_var_with_nested_key(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        _write_json(cfg_file, {})

        env = {"EASYWEAVER_DATABASE.HOST": "myhost"}
        with patch.dict(os.environ, env, clear=False):
            loader = ConfigurationLoader(config_path=str(tmp_path))

        assert loader.configuration.get("database", {}).get("host") == "myhost"

    def test_custom_config_file_name(self, tmp_path):
        cfg_file = tmp_path / "custom.json"
        _write_json(cfg_file, {"key": "value"})

        loader = ConfigurationLoader(config_path=str(tmp_path), config_file="custom.json")
        assert loader.configuration["key"] == "value"


# ---------------------------------------------------------------------------
# ConfigurationLoader._load_json_file
# ---------------------------------------------------------------------------


class TestLoadJsonFile:
    def test_returns_dict_from_valid_file(self, tmp_path):
        f = tmp_path / "data.json"
        _write_json(f, {"a": 1})
        result = ConfigurationLoader._load_json_file(str(f))
        assert result == {"a": 1}

    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        result = ConfigurationLoader._load_json_file(str(tmp_path / "missing.json"))
        assert result == {}


# ---------------------------------------------------------------------------
# ConfigurationLoader._flatten_to_dot_paths
# ---------------------------------------------------------------------------


class TestFlattenToDotPaths:
    def test_flat_dict_unchanged(self):
        result = ConfigurationLoader._flatten_to_dot_paths({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict_flattened(self):
        result = ConfigurationLoader._flatten_to_dot_paths({"a": {"b": {"c": 42}}})
        assert result == {"a.b.c": 42}

    def test_mixed_nested(self):
        result = ConfigurationLoader._flatten_to_dot_paths({"host": "h", "db": {"name": "n", "port": 5}})
        assert result["host"] == "h"
        assert result["db.name"] == "n"
        assert result["db.port"] == 5

    def test_non_dict_input_returns_empty(self):
        result = ConfigurationLoader._flatten_to_dot_paths([1, 2, 3])
        assert result == {}

    def test_with_parent_prefix(self):
        result = ConfigurationLoader._flatten_to_dot_paths({"x": 1}, parent="root")
        assert result == {"root.x": 1}


# ---------------------------------------------------------------------------
# ConfigurationLoader._resolve_placeholders
# ---------------------------------------------------------------------------


class TestResolvePlaceholders:
    def test_exact_placeholder_replaced_with_typed_value(self):
        result = ConfigurationLoader._resolve_placeholders(
            "{host}", {"host": "myhost"}
        )
        assert result == "myhost"

    def test_exact_placeholder_typed_as_int(self):
        result = ConfigurationLoader._resolve_placeholders("{port}", {"port": 5432})
        assert result == 5432

    def test_embedded_placeholder_string_substituted(self):
        result = ConfigurationLoader._resolve_placeholders(
            "connect to {host}:{port}", {"host": "db", "port": 5432}
        )
        assert result == "connect to db:5432"

    def test_unknown_placeholder_left_as_is(self):
        result = ConfigurationLoader._resolve_placeholders("{unknown}", {})
        assert result == "{unknown}"

    def test_nested_dict_resolved(self):
        config = {"db": {"host": "{host}"}}
        result = ConfigurationLoader._resolve_placeholders(config, {"host": "myhost"})
        assert result["db"]["host"] == "myhost"

    def test_list_items_resolved(self):
        config = ["{a}", "{b}"]
        result = ConfigurationLoader._resolve_placeholders(config, {"a": 1, "b": 2})
        assert result == [1, 2]

    def test_non_string_passthrough(self):
        result = ConfigurationLoader._resolve_placeholders(42, {})
        assert result == 42

    def test_dict_value_embedded_serialised_as_json(self):
        result = ConfigurationLoader._resolve_placeholders(
            "data={info}", {"info": {"k": "v"}}
        )
        assert result == 'data={"k": "v"}'


# ---------------------------------------------------------------------------
# ConfigurationLoader._extract_placeholder_keys
# ---------------------------------------------------------------------------


class TestExtractPlaceholderKeys:
    def test_extracts_from_string(self):
        keys = ConfigurationLoader._extract_placeholder_keys("{host}")
        assert "host" in keys

    def test_extracts_from_nested_dict(self):
        keys = ConfigurationLoader._extract_placeholder_keys({"a": {"b": "{port}"}})
        assert "port" in keys

    def test_extracts_from_list(self):
        keys = ConfigurationLoader._extract_placeholder_keys(["{x}", "{y}"])
        assert "x" in keys
        assert "y" in keys

    def test_multiple_configs(self):
        keys = ConfigurationLoader._extract_placeholder_keys({"a": "{x}"}, "{y}")
        assert "x" in keys
        assert "y" in keys

    def test_no_placeholders_returns_empty(self):
        keys = ConfigurationLoader._extract_placeholder_keys({"a": "plain"})
        assert len(keys) == 0


# ---------------------------------------------------------------------------
# ConfigurationLoader._collect_unresolved
# ---------------------------------------------------------------------------


class TestCollectUnresolved:
    def test_no_placeholders_returns_empty(self):
        result = ConfigurationLoader._collect_unresolved({"host": "myhost"})
        assert result == []

    def test_unresolved_placeholder_detected(self):
        result = ConfigurationLoader._collect_unresolved({"host": "{host}"})
        assert len(result) == 1
        assert "host" in result[0]

    def test_nested_unresolved(self):
        result = ConfigurationLoader._collect_unresolved({"db": {"host": "{host}"}})
        assert len(result) == 1

    def test_list_unresolved(self):
        result = ConfigurationLoader._collect_unresolved({"tags": ["{t1}", "plain"]})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# ConfigurationLoader._convert_value
# ---------------------------------------------------------------------------


class TestConvertValue:
    def setup_method(self):
        self.loader = ConfigurationLoader.__new__(ConfigurationLoader)
        from easyweaver.utils.dict_util import DictUtil
        self.loader._dict_util = DictUtil()
        self.loader.configuration = {}

    def test_converts_int_string(self):
        assert self.loader._convert_value("42") == 42

    def test_converts_float_string(self):
        assert self.loader._convert_value("3.14") == pytest.approx(3.14)

    def test_converts_true_bool(self):
        assert self.loader._convert_value("true") is True

    def test_converts_false_bool(self):
        assert self.loader._convert_value("false") is False

    def test_converts_json_object(self):
        result = self.loader._convert_value('{"a": 1}')
        assert result == {"a": 1}

    def test_converts_json_list(self):
        result = self.loader._convert_value('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_returns_string_for_plain_text(self):
        assert self.loader._convert_value("hello") == "hello"

    def test_converts_ruby_hash(self):
        result = self.loader._convert_value('{:host=>"myhost", :port=>"5432"}')
        assert isinstance(result, dict)
        assert result.get("host") == "myhost"


# ---------------------------------------------------------------------------
# ConfigurationLoader.get_config_by_path
# ---------------------------------------------------------------------------


class TestGetConfigByPath:
    def setup_method(self):
        self.loader = ConfigurationLoader.__new__(ConfigurationLoader)
        from easyweaver.utils.dict_util import DictUtil
        self.loader._dict_util = DictUtil()
        self.loader.configuration = {
            "databases": {
                "main": {
                    "db_info": {"host": "db-host", "port": 5432}
                }
            },
            "simple": "value",
        }
        self.loader.unresolved_properties = []

    def test_gets_top_level_value(self):
        assert self.loader.get_config_by_path("simple") == "value"

    def test_gets_nested_value(self):
        result = self.loader.get_config_by_path("databases.main.db_info.host")
        assert result == "db-host"

    def test_returns_default_for_missing_path(self):
        assert self.loader.get_config_by_path("nonexistent", default="fallback") == "fallback"

    def test_returns_none_for_missing_path_no_default(self):
        assert self.loader.get_config_by_path("nonexistent") is None


# ---------------------------------------------------------------------------
# ConfigurationLoader.get_db_config
# ---------------------------------------------------------------------------


class TestGetDBConfig:
    def setup_method(self):
        self.loader = ConfigurationLoader.__new__(ConfigurationLoader)
        from easyweaver.utils.dict_util import DictUtil
        self.loader._dict_util = DictUtil()
        self.loader.unresolved_properties = []

    def test_get_db_config_with_db_info_key(self):
        self.loader.configuration = {
            "databases": {
                "main": {"db_info": {"host": "h", "port": 5432}}
            }
        }
        result = self.loader.get_db_config("main")
        assert result["host"] == "h"

    def test_get_db_config_with_dot_info_key(self):
        self.loader.configuration = {
            "databases": {
                "main": {"db.info": {"host": "h2", "port": 5433}}
            }
        }
        result = self.loader.get_db_config("main")
        assert result["host"] == "h2"

    def test_get_db_config_flat_structure(self):
        self.loader.configuration = {
            "databases": {
                "main": {"host": "flat-host"}
            }
        }
        result = self.loader.get_db_config("main")
        assert result["host"] == "flat-host"

    def test_get_db_config_missing_returns_none(self):
        self.loader.configuration = {"databases": {}}
        assert self.loader.get_db_config("nonexistent") is None

    def test_get_config_by_token(self):
        self.loader.configuration = {
            "databases": {
                "auth": {"host": "auth-db"}
            }
        }
        result = self.loader.get_config_by_token("auth")
        assert result["host"] == "auth-db"


# ---------------------------------------------------------------------------
# ConfigurationLoader._set_nested_value
# ---------------------------------------------------------------------------


class TestSetNestedValue:
    def setup_method(self):
        self.loader = ConfigurationLoader.__new__(ConfigurationLoader)
        from easyweaver.utils.dict_util import DictUtil
        self.loader._dict_util = DictUtil()
        self.loader.configuration = {}
        self.loader.unresolved_properties = []

    def test_sets_top_level(self):
        self.loader._set_nested_value("key", "value")
        assert self.loader.configuration["key"] == "value"

    def test_sets_nested_path(self):
        self.loader._set_nested_value("a.b.c", 42)
        assert self.loader.configuration["a"]["b"]["c"] == 42

    def test_overwrites_scalar_with_dict(self):
        self.loader.configuration["a"] = "scalar"
        self.loader._set_nested_value("a.b", "nested")
        assert isinstance(self.loader.configuration["a"], dict)
        assert self.loader.configuration["a"]["b"] == "nested"


# ---------------------------------------------------------------------------
# ConfigurationLoader._collect_env_overrides
# ---------------------------------------------------------------------------


class TestCollectEnvOverrides:
    def setup_method(self):
        self.loader = ConfigurationLoader.__new__(ConfigurationLoader)
        from easyweaver.utils.dict_util import DictUtil
        self.loader._dict_util = DictUtil()
        self.loader.configuration = {}
        self.loader.unresolved_properties = []
        self.loader.env_prefix = "EASYWEAVER"

    def test_picks_up_matching_env_var(self):
        known = {"host"}
        env = {"EASYWEAVER_HOST": "myhost"}
        overrides = self.loader._collect_env_overrides(known, "EASYWEAVER", env_dict=env)
        assert overrides.get("host") == "myhost"

    def test_ignores_non_matching_prefix(self):
        known = {"host"}
        env = {"OTHER_HOST": "other"}
        overrides = self.loader._collect_env_overrides(known, "EASYWEAVER", env_dict=env)
        assert "host" not in overrides

    def test_skips_environment_meta_key(self):
        known = set()
        env = {"EASYWEAVER_ENVIRONMENT": "dev"}
        overrides = self.loader._collect_env_overrides(known, "EASYWEAVER", env_dict=env)
        assert "environment" not in overrides

    def test_dot_prefix_form_accepted(self):
        known = {"db.host"}
        env = {"EASYWEAVER.DB.HOST": "db-host"}
        overrides = self.loader._collect_env_overrides(known, "EASYWEAVER", env_dict=env)
        assert "db.host" in overrides

    def test_json_value_expanded_to_flat_paths(self):
        known = {"db"}
        env = {"EASYWEAVER_DB": '{"host": "h", "port": 5432}'}
        overrides = self.loader._collect_env_overrides(known, "EASYWEAVER", env_dict=env)
        # The dict should be stored at "db" and also flattened
        assert isinstance(overrides.get("db"), dict)
        assert overrides.get("db.host") == "h"


# ---------------------------------------------------------------------------
# ConfigurationLoader.load_environment (full pipeline)
# ---------------------------------------------------------------------------


class TestLoadEnvironment:
    def _write_config_files(self, tmp_path, env="dev"):
        config = {"database": {"host": "{db.host}", "port": "{db.port}"}}
        localenv = {"db.host": "localhost", "db.port": 5432}
        env_addon = {}
        _write_json(tmp_path / "config.json", config)
        _write_json(tmp_path / f"localenv-{env}.json", localenv)
        _write_json(tmp_path / f"{env}.json", env_addon)

    def test_basic_pipeline_resolves_placeholders(self, tmp_path):
        self._write_config_files(tmp_path)
        loader = ConfigurationLoader(
            config_path=str(tmp_path), environment="dev"
        )
        assert loader.configuration["database"]["host"] == "localhost"
        assert loader.configuration["database"]["port"] == 5432

    def test_os_env_vars_override_localenv(self, tmp_path):
        self._write_config_files(tmp_path)
        env = {"EASYWEAVER_DB.HOST": "override-host"}
        with patch.dict(os.environ, env, clear=False):
            loader = ConfigurationLoader(
                config_path=str(tmp_path), environment="dev"
            )
        assert loader.configuration["database"]["host"] == "override-host"

    def test_unresolved_placeholders_logged(self, tmp_path):
        config = {"key": "{missing_ref}"}
        localenv = {}
        _write_json(tmp_path / "config.json", config)
        _write_json(tmp_path / "localenv-dev.json", localenv)
        _write_json(tmp_path / "dev.json", {})

        loader = ConfigurationLoader(config_path=str(tmp_path), environment="dev")
        assert len(loader.unresolved_properties) > 0

    def test_missing_files_result_in_partial_config(self, tmp_path):
        # Only write config.json; localenv and env files are missing
        _write_json(tmp_path / "config.json", {"key": "value"})
        loader = ConfigurationLoader(config_path=str(tmp_path), environment="dev")
        # Should not raise; config may have unresolved placeholders
        assert isinstance(loader.configuration, dict)

    def test_env_addon_keys_merged_into_config(self, tmp_path):
        _write_json(tmp_path / "config.json", {"existing": "yes"})
        _write_json(tmp_path / "localenv-dev.json", {})
        _write_json(tmp_path / "dev.json", {"extra_key": "extra_value"})
        loader = ConfigurationLoader(config_path=str(tmp_path), environment="dev")
        assert loader.configuration.get("extra_key") == "extra_value"

    def test_simulator_file_sets_defaults(self, tmp_path):
        sim_file = tmp_path / "simulator.json"
        _write_json(sim_file, {"db.host": "sim-host"})
        _write_json(tmp_path / "config.json", {"host": "{db.host}"})
        _write_json(tmp_path / "localenv-dev.json", {})
        _write_json(tmp_path / "dev.json", {})

        loader = ConfigurationLoader(
            config_path=str(tmp_path),
            environment="dev",
            simulator_file=str(sim_file),
        )
        assert loader.configuration["host"] == "sim-host"

        # Cleanup env var set by simulator
        env_key = "EASYWEAVER_DB.HOST"
        os.environ.pop(env_key, None)


# ---------------------------------------------------------------------------
# ConfigValueSimulator
# ---------------------------------------------------------------------------


class TestConfigValueSimulator:
    def test_load_simulator_file_returns_data(self, tmp_path):
        sim = tmp_path / "sim.json"
        _write_json(sim, {"db.host": "h", "db.port": 5432})

        result = ConfigValueSimulator.load_simulator_file(str(sim))
        assert result["db.host"] == "h"
        assert result["db.port"] == 5432

        # Cleanup env vars
        os.environ.pop("EASYWEAVER_DB.HOST", None)
        os.environ.pop("EASYWEAVER_DB.PORT", None)

    def test_load_simulator_sets_env_vars(self, tmp_path):
        sim = tmp_path / "sim.json"
        _write_json(sim, {"my.key": "myval"})

        ConfigValueSimulator.load_simulator_file(str(sim))
        assert os.environ.get("EASYWEAVER_MY.KEY") == "myval"

        os.environ.pop("EASYWEAVER_MY.KEY", None)

    def test_load_simulator_does_not_overwrite_existing_env(self, tmp_path):
        sim = tmp_path / "sim.json"
        _write_json(sim, {"x.key": "from-sim"})
        os.environ["EASYWEAVER_X.KEY"] = "from-env"

        ConfigValueSimulator.load_simulator_file(str(sim))
        assert os.environ["EASYWEAVER_X.KEY"] == "from-env"

        os.environ.pop("EASYWEAVER_X.KEY", None)

    def test_load_simulator_missing_file_returns_empty(self, tmp_path):
        result = ConfigValueSimulator.load_simulator_file(str(tmp_path / "missing.json"))
        assert result == {}

    def test_load_simulator_serialises_dict_values(self, tmp_path):
        sim = tmp_path / "sim.json"
        _write_json(sim, {"creds": {"user": "u", "pass": "p"}})

        ConfigValueSimulator.load_simulator_file(str(sim))
        env_val = os.environ.get("EASYWEAVER_CREDS")
        assert env_val is not None
        parsed = json.loads(env_val)
        assert parsed["user"] == "u"

        os.environ.pop("EASYWEAVER_CREDS", None)

    def test_set_os_environment_sets_vars(self):
        values = {"db": {"host": "testhost", "port": 9999}}
        ConfigValueSimulator.set_os_environment(values, prefix="TESTPFX")

        assert os.environ.get("TESTPFX_DB.HOST") == "testhost"
        assert os.environ.get("TESTPFX_DB.PORT") == "9999"

        os.environ.pop("TESTPFX_DB.HOST", None)
        os.environ.pop("TESTPFX_DB.PORT", None)

    def test_load_simulator_serialises_bool_values(self, tmp_path):
        sim = tmp_path / "sim.json"
        _write_json(sim, {"flag": True})

        ConfigValueSimulator.load_simulator_file(str(sim))
        env_val = os.environ.get("EASYWEAVER_FLAG")
        assert env_val == "true"

        os.environ.pop("EASYWEAVER_FLAG", None)
