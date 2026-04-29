"""Configuration Loader"""
import os
import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from copy import deepcopy

from .dict_util import DictUtil, parse_ruby_hash
from easyweaver import ENVIRONEMNT_VARIABLE_PREFIX, OS_PROPERTY_SEPRATOR

logger = logging.getLogger(__name__)


class ConfigurationLoader:
    """Configuration loader from JSON files and environment variables"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_file: str = "config.json",
        env_prefix: str = ENVIRONEMNT_VARIABLE_PREFIX,
        environment: Optional[str] = None,
        simulator_file: Optional[str] = None
    ):
        self.config_path = config_path or os.getcwd()
        self.env_prefix = env_prefix
        self.configuration: Dict[str, Any] = {}
        self.unresolved_properties: List[str] = []
        self._dict_util = DictUtil()

        if environment:
            self.load_environment(self.config_path, environment, simulator_file)
        else:
            # Legacy mode: load single config file + env vars
            self._load_config(config_file)
            self._apply_env_vars()

    @staticmethod
    def _load_json_file(file_path: str) -> Dict[str, Any]:
        """Load JSON file, return empty dict if missing or invalid"""
        p = Path(file_path)
        if not p.exists():
            return {}
        with open(p, 'r') as f:
            return json.load(f)

    @staticmethod
    def _flatten_to_dot_paths(d: Any, parent: str = "") -> Dict[str, Any]:
        """Flatten nested dict to {\"a.b.c\": value} dot-path lookup"""
        sep = OS_PROPERTY_SEPRATOR
        items: Dict[str, Any] = {}
        if not isinstance(d, dict):
            return items
        for key, value in d.items():
            new_key = f"{parent}{sep}{key}" if parent else key
            if isinstance(value, dict):
                items.update(ConfigurationLoader._flatten_to_dot_paths(value, new_key))
            else:
                items[new_key] = value
        return items

    @staticmethod
    def _resolve_placeholders(config: Any, values_lookup: Dict[str, Any]) -> Any:
        """Recursively resolve {dot.path} placeholders in config.

        - If a string is exactly \"{dot.path}\", replace with typed value.
        - If a string contains {dot.path} embedded, do string substitution.
        - If no match found, keep original string (unresolved).
        """
        if isinstance(config, dict):
            return {
                k: ConfigurationLoader._resolve_placeholders(v, values_lookup)
                for k, v in config.items()
            }
        elif isinstance(config, list):
            return [
                ConfigurationLoader._resolve_placeholders(item, values_lookup)
                for item in config
            ]
        elif isinstance(config, str):
            # Exact match: entire string is a single placeholder
            exact_match = re.fullmatch(r'\{([^{}]+)\}', config)
            if exact_match:
                dot_path = exact_match.group(1)
                if dot_path in values_lookup:
                    return deepcopy(values_lookup[dot_path])
                return config

            # Embedded placeholders: string contains one or more {dot.path}
            def replace_match(m):
                dot_path = m.group(1)
                if dot_path in values_lookup:
                    val = values_lookup[dot_path]
                    if isinstance(val, (dict, list)):
                        return json.dumps(val)
                    return str(val)
                return m.group(0)

            resolved = re.sub(r'\{([^{}]+)\}', replace_match, config)
            return resolved
        else:
            return config

    @staticmethod
    def _extract_placeholder_keys(*configs: Any) -> set:
        """Extract all {dot.path} placeholder keys from config structures."""
        keys: set = set()

        def _scan(obj: Any) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    _scan(v)
            elif isinstance(obj, list):
                for item in obj:
                    _scan(item)
            elif isinstance(obj, str):
                for m in re.finditer(r'\{([^{}]+)\}', obj):
                    keys.add(m.group(1))

        for config in configs:
            _scan(config)
        return keys

    @staticmethod
    def _collect_unresolved(config: Any, parent: str = "") -> List[str]:
        """Collect dot-paths of values that still contain {placeholder} strings."""
        sep = OS_PROPERTY_SEPRATOR
        unresolved: List[str] = []
        if isinstance(config, dict):
            for key, value in config.items():
                path = f"{parent}{sep}{key}" if parent else key
                unresolved.extend(
                    ConfigurationLoader._collect_unresolved(value, path)
                )
        elif isinstance(config, list):
            for i, item in enumerate(config):
                path = f"{parent}[{i}]"
                unresolved.extend(
                    ConfigurationLoader._collect_unresolved(item, path)
                )
        elif isinstance(config, str) and re.search(r'\{[^{}]+\}', config):
            unresolved.append(f"{parent} = {config}")
        return unresolved

    @staticmethod
    def _normalise_env_key(key: str) -> str:
        """Upper-case the env var key for case-insensitive matching.

        Dots and underscores are preserved as-is because dots are path
        separators (OS_PROPERTY_SEPRATOR) while underscores are legitimate
        parts of property names (e.g. ``db_info``, ``connection_scheme``).
        """
        return key.upper()

    def _collect_env_overrides(
        self, known_dot_paths: set, prefix: str,
        env_dict: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Scan OS env vars with *prefix* and map back to dot-path keys.

        Env vars are expected to use dots as path separators (matching
        ``OS_PROPERTY_SEPRATOR``), e.g.::

            EASYLIFE_DATABASES.AUTHENTICATION.DB_INFO.HOST=myhost
            EASYLIFE.ENVIRONMENT.SMTP.SMTP_SERVER=mailpit

        Underscores within a segment are preserved (``db_info`` stays
        ``db_info``).  Both ``PREFIX_`` and ``PREFIX.`` prefixes are accepted.
        """
        sep = OS_PROPERTY_SEPRATOR
        prefix_us = f"{prefix}_"          # EASYLIFE_
        prefix_dot = f"{prefix}{sep}"     # EASYLIFE.

        # Build reverse map: UPPER-CASED env key → original dot-path.
        # Register both prefix forms so either matches.
        reverse_map: Dict[str, str] = {}
        for prop_path in known_dot_paths:
            key_us = f"{prefix_us}{prop_path}".upper()
            key_dot = f"{prefix_dot}{prop_path}".upper()
            reverse_map[key_us] = prop_path
            reverse_map[key_dot] = prop_path

        # Meta env vars to skip (not config values)
        skip_keys: set = set()
        for meta_suffix in ["ENVIRONMENT"]:
            skip_keys.add(f"{prefix_us}{meta_suffix}".upper())
            skip_keys.add(f"{prefix_dot}{meta_suffix}".upper())
        skip_keys.add("CONFIG_PATH")

        overrides: Dict[str, Any] = {}
        # Both prefix forms have the same length (prefix + 1 separator char)
        prefix_len = len(prefix_us)
        source = env_dict if env_dict is not None else os.environ
        for key, value in source.items():
            if not (key.startswith(prefix_us) or key.startswith(prefix_dot)):
                continue
            upper_key = key.upper()
            if upper_key in skip_keys:
                continue
            if upper_key in reverse_map:
                prop_path = reverse_map[upper_key]
            else:
                # Fallback: strip prefix, lowercase — dots stay as separators,
                # underscores stay as part of property names.
                suffix = key[prefix_len:]
                prop_path = suffix.lower()
            converted = self._convert_value(value)
            overrides[prop_path] = converted
            # If the converted value is a dict, also add flattened dot-path
            # entries so individual field placeholders can be resolved.
            if isinstance(converted, dict):
                overrides.update(
                    self._flatten_to_dot_paths(converted, prop_path)
                )
        return overrides

    def load_environment(
        self,
        config_path: str,
        environment: str,
        simulator_file: Optional[str] = None
    ) -> None:
        """Main config pipeline — 3 files + OS env vars.

        Files (required):
          1. localenv-{env}.json  — base property values (flat or nested)
          2. {env}.json           — env-specific addon config with {placeholder} refs
          3. config.json          — base app config with {placeholder} refs

        Optional:
          simulator_file — path to a flat JSON file that sets env var defaults
                           for local dev.  Only loaded when explicitly provided.

        Priority (lowest → highest):
          simulator defaults → localenv hardcoded → OS env vars
        """
        base = Path(config_path)

        # a) Optionally load simulator file (only when explicitly provided)
        simulator_data: Dict[str, Any] = {}
        if simulator_file:
            pre_sim_env = dict(os.environ)
            simulator_data = ConfigValueSimulator.load_simulator_file(
                simulator_file, self.env_prefix
            )
        else:
            pre_sim_env = None  # no snapshot needed

        # b) Load all template files (raw, unresolved)
        localenv_path = base / f"localenv-{environment}.json"
        localenv_raw = self._load_json_file(str(localenv_path))
        env_path = base / f"{environment}.json"
        env_raw = self._load_json_file(str(env_path))
        config_json_path = base / "config.json"
        config_raw = self._load_json_file(str(config_json_path))

        # c) Collect all known dot-path keys for env var reverse mapping
        known_dot_paths = set(simulator_data.keys())
        known_dot_paths.update(
            self._extract_placeholder_keys(localenv_raw, env_raw, config_raw)
        )
        known_dot_paths.update(
            self._flatten_to_dot_paths(localenv_raw).keys()
        )

        # d) Build values_lookup:
        #    simulator (lowest) → localenv hardcoded (overrides simulator)
        #    → OS env vars (highest, overrides everything)
        #    When a simulator file was loaded, use the pre-simulator env
        #    snapshot so simulator-set vars don't count as OS overrides.
        localenv_flat = self._flatten_to_dot_paths(localenv_raw)
        env_overrides = self._collect_env_overrides(
            known_dot_paths, self.env_prefix, pre_sim_env
        )

        values_lookup: Dict[str, Any] = {}
        values_lookup.update(simulator_data)       # simulator defaults (lowest)
        # Only override simulator with localenv values that are NOT placeholders
        localenv_resolved_values = {
            k: v for k, v in localenv_flat.items()
            if not (isinstance(v, str) and v.strip().startswith('{') and v.strip().endswith('}'))
        }
        values_lookup.update(localenv_resolved_values)  # localenv hardcoded override simulator
        values_lookup.update(env_overrides)         # OS env vars (highest)

        # e) Resolve any {placeholders} within localenv itself, re-flatten
        localenv_config = self._resolve_placeholders(localenv_raw, values_lookup)
        localenv_flat_resolved = self._flatten_to_dot_paths(localenv_config)
        values_lookup.update(localenv_flat_resolved)

        # f) Resolve {environment}.json from values dict, then re-apply env
        #    overrides so OS vars always win over both localenv and env_config
        env_config = self._resolve_placeholders(env_raw, values_lookup)
        env_flat = self._flatten_to_dot_paths(env_config)
        values_lookup.update(env_flat)
        values_lookup.update(env_overrides)         # OS env vars still win

        # g) Resolve config.json with final values lookup
        resolved_config = self._resolve_placeholders(config_raw, values_lookup)

        # h) Merge extra properties from env_config into resolved_config
        #    (adds env-specific keys not present in config.json)
        self._dict_util.merge_dicts(resolved_config, env_config)

        # i) Collect unresolved placeholders
        self.unresolved_properties = sorted(
            self._collect_unresolved(resolved_config)
        )
        if self.unresolved_properties:
            logger.warning(
                "Unresolved config placeholders (%d): %s",
                len(self.unresolved_properties),
                ", ".join(self.unresolved_properties),
            )

        # j) Set final configuration
        self.configuration = resolved_config

    def _load_config(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        file_path = Path(self.config_path) / config_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                self.configuration = json.load(f)

    def _apply_env_vars(self) -> None:
        """Apply environment variables to configuration"""
        prefix = f"{self.env_prefix}_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._set_nested_value(config_key, self._convert_value(value))

    def _set_nested_value(self, key_path: str, value: Any) -> None:
        """Set a nested value in configuration"""
        keys = key_path.split(OS_PROPERTY_SEPRATOR)
        d = self.configuration
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            elif not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        # Try JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try Ruby hash (e.g. GCS credentials_json from env vars)
        ruby_result = parse_ruby_hash(value)
        if ruby_result is not None:
            return ruby_result

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'

        return value

    def get_config_by_path(
        self,
        key_path: str,
        default: Any = None
    ) -> Any:
        """Get configuration value by dot-notation path"""
        result = self._dict_util.get_deep_nested_value(
            self.configuration,
            key_path,
            default
        )
        return result if result is not None else default

    def get_db_config(self, token: str) -> Optional[Dict[str, Any]]:
        """Get database configuration by token"""
        config = self.get_config_by_path(f"databases.{token}")
        if config and 'db.info' in config:
            # Handle both db_info and db.info formats
            return deepcopy(config['db.info'])
        if config and 'db_info' in config:
            return deepcopy(config['db_info'])
        # Check for flat structure from env vars
        if config:
            return deepcopy(config)
        return None

    def get_config_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Get configuration by token"""
        return self.get_config_by_path(f"databases.{token}")


class ConfigValueSimulator:
    """Simulate configuration from environment variables for testing"""

    @staticmethod
    def load_simulator_file(
        file_path: str,
        prefix: str = ENVIRONEMNT_VARIABLE_PREFIX
    ) -> Dict[str, Any]:
        """Load simulator JSON file and set env vars.
        File format: flat dict {\"dot.path.key\": value, ...}
        Each entry becomes env var PREFIX_DOT_PATH_KEY=value
        Returns raw flat dict for use as values lookup.
        """
        p = Path(file_path)
        if not p.exists():
            return {}
        with open(p, 'r') as f:
            simulator_data = json.load(f)

        for dot_path, value in simulator_data.items():
            env_key = f"{prefix}_{dot_path}".upper()
            # Only set if not already present — existing env vars (e.g. from
            # Docker) take priority over simulator defaults.
            if env_key not in os.environ:
                if isinstance(value, (dict, list)):
                    os.environ[env_key] = json.dumps(value)
                elif isinstance(value, bool):
                    os.environ[env_key] = str(value).lower()
                else:
                    os.environ[env_key] = str(value)

        return simulator_data

    @staticmethod
    def set_os_environment(
        values: Dict[str, Any],
        prefix: str = ENVIRONEMNT_VARIABLE_PREFIX
    ) -> None:
        """Set environment variables from dictionary"""
        sep = OS_PROPERTY_SEPRATOR

        def flatten(d: Dict, parent_key: str = '') -> Dict[str, str]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key).items())
                else:
                    items.append((new_key, str(v) if not isinstance(v, (dict, list)) else json.dumps(v)))
            return dict(items)

        flat_values = flatten(values)
        for key, value in flat_values.items():
            env_key = f"{prefix}_{key}".upper()
            os.environ[env_key] = value
