"""Dictionary Utility"""
import re
from functools import reduce
from typing import Any, Dict, Optional

from easyweaver import OS_PROPERTY_SEPRATOR


def parse_ruby_hash(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a Ruby hash string into a Python dict.

    Ruby hash format: {:key=>"value", key2=>"value2"}
    Keys may have an optional ``:`` prefix (Ruby symbol notation).
    Values are double-quoted strings.

    Returns a dict on success, or ``None`` if *raw* does not look like a
    Ruby hash (allowing callers to fall through to other converters).
    """
    s = raw.strip()
    if not (s.startswith("{") and s.endswith("}") and "=>" in s):
        return None

    inner = s[1:-1]

    # :?key => "value"  (key may contain dots, dashes, or word chars)
    pattern = re.compile(
        r""":?([\w.\-]+)\s*=>\s*"((?:[^"\\]|\\.)*)" """.strip(),
    )

    result: Dict[str, Any] = {}
    for match in pattern.finditer(inner):
        key = match.group(1)
        value = match.group(2)
        # Certificate/PEM values pass through as-is
        if "-----BEGIN" not in value:
            # Unescape common Ruby string escape sequences
            value = (
                value
                .replace('\\"', '"')
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
            )
        result[key] = value

    return result if result else None


class DictUtil:
    """Dictionary utility class for deep nested operations"""

    def deep_get(self, dictionary: Dict, keys: str, default: Any = None) -> Any:
        """Get value from nested dictionary using dot notation"""
        return reduce(
            lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
            keys.split(OS_PROPERTY_SEPRATOR),
            dictionary
        )

    def get_deep_nested_value(
        self,
        dictionary: Dict,
        key_path: str,
        default: Any = None
    ) -> Any:
        """Get deeply nested value with list support"""
        keys = key_path.split(OS_PROPERTY_SEPRATOR)

        def _step(d: Any, key: str) -> Any:
            if isinstance(d, list):
                return [
                    item.get(key)
                    for sublist in d
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                    if isinstance(item, dict)
                ]
            if isinstance(d, dict):
                return d.get(key)
            return default

        try:
            return reduce(_step, keys, dictionary)
        except AttributeError:
            return None

    def merge_dicts(self, config_dict: Dict, setting_dict: Dict) -> None:
        """Merge two dictionaries recursively"""
        for key, value in setting_dict.items():
            if key not in config_dict:
                config_dict[key] = value
            else:
                if isinstance(value, dict) and isinstance(config_dict[key], dict):
                    self.merge_dicts(config_dict[key], value)
                elif isinstance(value, list) and isinstance(config_dict[key], list):
                    config_dict[key].extend(x for x in value if x not in config_dict[key])
                else:
                    config_dict[key] = value


class DictUtilError(Exception):
    """Dictionary utility error"""
    pass
