"""URL Arguments Utility"""
import re
import json
from typing import Any, Dict, List, Union

integer_pattern = re.compile(r'^-?(0|[1-9]\d*)$')
float_pattern = re.compile(r'^-?\d*\.\d+$')


def convert_str(value: str, key: str = None) -> Any:
    """Convert string to appropriate type"""
    try:
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        elif "." in value:
            return float(value)
        elif bool(integer_pattern.match(value)):
            if key is not None and "query_" in key:
                return value
            else:
                return int(value)
        elif bool(float_pattern.match(value)):
            return float(value)
        else:
            return value
    except ValueError:
        return value


def is_stringified_json(value: Any) -> bool:
    """Check if value is a stringified JSON"""
    if not isinstance(value, str):
        return False
    try:
        result = json.loads(value)
        return isinstance(result, (list, dict))
    except json.JSONDecodeError:
        return False


def get_stringified_json_to_dict(value: Any, key: str = None) -> Any:
    """Parse stringified JSON or convert string"""
    if is_stringified_json(value):
        return json.loads(value)
    return convert_str(value, key) if isinstance(value, str) else value


def is_valid_value_list(input_list: List) -> bool:
    """Check if list contains only valid types"""
    valid_types = (str, int, float, bool)
    return all(isinstance(item, valid_types) for item in input_list)


def parse_key_path(key: str) -> List[Union[str, int]]:
    """Parse key path with bracket notation"""
    pattern = r'\w+|\[\w+\]'
    parts = re.findall(pattern, key)
    result = []
    for part in parts:
        part = part.strip('[]')
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part)
    return result


def assign_nested(container: Union[Dict, List], path: List, value: Any) -> None:
    """Assign value to nested structure"""
    for i, part in enumerate(path):
        is_last = i == len(path) - 1
        if isinstance(part, int):
            while len(container) <= part:
                container.append(None)
            if is_last:
                container[part] = value
            else:
                next_part = path[i + 1]
                if container[part] is None:
                    container[part] = [] if isinstance(next_part, int) else {}
                container = container[part]
        else:
            if part not in container:
                next_part = path[i + 1] if not is_last else None
                container[part] = [] if isinstance(next_part, int) else {}
            if is_last:
                container[part] = value
            else:
                container = container[part]


def parse_url_args(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Parse flat URL arguments to nested structure"""
    result = {}
    for key, value in flat_dict.items():
        path = parse_key_path(key)
        if is_stringified_json(value):
            value = get_stringified_json_to_dict(value, key)
        assign_nested(result, path, value)
    return result


def clean_parse_url_args(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and parse URL arguments from FastAPI query params"""
    collected_kwargs = {}
    
    for key, value in params.items():
        if isinstance(value, list):
            if len(value) > 1:
                if not is_valid_value_list(value):
                    p_values = [get_stringified_json_to_dict(v) for v in value]
                    p_list = [i if isinstance(i, list) else [i] for i in p_values]
                    flatten_list = [i for sublist in p_list for i in sublist]
                    collected_kwargs[key] = flatten_list
                else:
                    collected_kwargs[key] = value
            else:
                collected_kwargs[key] = get_stringified_json_to_dict(value[0], key)
        else:
            collected_kwargs[key] = get_stringified_json_to_dict(value, key)
    
    return parse_url_args(collected_kwargs)
