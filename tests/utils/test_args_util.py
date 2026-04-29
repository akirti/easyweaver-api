"""Tests for easyweaver.utils.args_util"""
import json
import pytest

from easyweaver.utils.args_util import (
    assign_nested,
    clean_parse_url_args,
    convert_str,
    get_stringified_json_to_dict,
    is_stringified_json,
    is_valid_value_list,
    parse_key_path,
    parse_url_args,
)


# ---------------------------------------------------------------------------
# convert_str
# ---------------------------------------------------------------------------


class TestConvertStr:
    def test_true_string(self):
        assert convert_str("true") is True

    def test_false_string(self):
        assert convert_str("false") is False

    def test_true_uppercase(self):
        assert convert_str("True") is True

    def test_false_mixed_case(self):
        assert convert_str("FALSE") is False

    def test_integer_string(self):
        assert convert_str("42") == 42
        assert isinstance(convert_str("42"), int)

    def test_negative_integer(self):
        assert convert_str("-7") == -7

    def test_zero(self):
        assert convert_str("0") == 0

    def test_float_string_with_dot(self):
        result = convert_str("3.14")
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_negative_float(self):
        result = convert_str("-1.5")
        assert isinstance(result, float)
        assert result == pytest.approx(-1.5)

    def test_plain_string(self):
        assert convert_str("hello") == "hello"

    def test_empty_string(self):
        assert convert_str("") == ""

    def test_query_key_keeps_integer_as_string(self):
        # When key contains "query_", integers must stay as strings
        result = convert_str("42", key="query_id")
        assert result == "42"
        assert isinstance(result, str)

    def test_non_query_key_converts_integer(self):
        result = convert_str("42", key="page_size")
        assert result == 42
        assert isinstance(result, int)

    def test_query_key_none_converts_integer(self):
        result = convert_str("10", key=None)
        assert result == 10

    def test_leading_zeros_not_integer(self):
        # "01" does not match integer pattern (leading zero)
        result = convert_str("01")
        assert result == "01"

    def test_float_pattern_no_leading_digit(self):
        # ".5" matches float_pattern
        result = convert_str(".5")
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# is_stringified_json
# ---------------------------------------------------------------------------


class TestIsStringifiedJson:
    def test_json_list_string(self):
        assert is_stringified_json('["a", "b"]') is True

    def test_json_dict_string(self):
        assert is_stringified_json('{"key": "value"}') is True

    def test_plain_string(self):
        assert is_stringified_json("hello") is False

    def test_json_number_string_not_valid(self):
        # A bare number is valid JSON but not list/dict
        assert is_stringified_json("42") is False

    def test_json_bool_string_not_valid(self):
        assert is_stringified_json("true") is False

    def test_non_string_input(self):
        assert is_stringified_json(42) is False
        assert is_stringified_json(None) is False
        assert is_stringified_json(["a"]) is False

    def test_empty_string(self):
        assert is_stringified_json("") is False

    def test_malformed_json(self):
        assert is_stringified_json("{not json}") is False

    def test_nested_json(self):
        assert is_stringified_json('{"a": {"b": 1}}') is True


# ---------------------------------------------------------------------------
# get_stringified_json_to_dict
# ---------------------------------------------------------------------------


class TestGetStringifiedJsonToDict:
    def test_stringified_list_parsed(self):
        result = get_stringified_json_to_dict('["x", "y"]')
        assert result == ["x", "y"]

    def test_stringified_dict_parsed(self):
        result = get_stringified_json_to_dict('{"k": "v"}')
        assert result == {"k": "v"}

    def test_plain_string_converted(self):
        result = get_stringified_json_to_dict("42")
        assert result == 42

    def test_non_string_passthrough(self):
        assert get_stringified_json_to_dict(99) == 99
        assert get_stringified_json_to_dict(None) is None
        assert get_stringified_json_to_dict([1, 2]) == [1, 2]

    def test_plain_string_with_key(self):
        result = get_stringified_json_to_dict("10", key="query_param")
        assert result == "10"  # kept as string because of query_ prefix


# ---------------------------------------------------------------------------
# is_valid_value_list
# ---------------------------------------------------------------------------


class TestIsValidValueList:
    def test_all_strings(self):
        assert is_valid_value_list(["a", "b", "c"]) is True

    def test_all_ints(self):
        assert is_valid_value_list([1, 2, 3]) is True

    def test_mixed_valid_types(self):
        assert is_valid_value_list([1, "hello", 3.14, True]) is True

    def test_empty_list(self):
        assert is_valid_value_list([]) is True

    def test_contains_dict(self):
        assert is_valid_value_list([{"key": "val"}]) is False

    def test_contains_list(self):
        assert is_valid_value_list([[1, 2]]) is False

    def test_contains_none(self):
        assert is_valid_value_list([None]) is False


# ---------------------------------------------------------------------------
# parse_key_path
# ---------------------------------------------------------------------------


class TestParseKeyPath:
    def test_simple_key(self):
        assert parse_key_path("name") == ["name"]

    def test_bracket_index(self):
        assert parse_key_path("items[0]") == ["items", 0]

    def test_nested_bracket(self):
        assert parse_key_path("a[0][1]") == ["a", 0, 1]

    def test_word_keys_only(self):
        assert parse_key_path("key") == ["key"]

    def test_bracket_string_key(self):
        # [word] where word is non-digit stays as string
        assert parse_key_path("obj[foo]") == ["obj", "foo"]

    def test_multiple_bracket_indices(self):
        result = parse_key_path("matrix[2][3]")
        assert result == ["matrix", 2, 3]


# ---------------------------------------------------------------------------
# assign_nested
# ---------------------------------------------------------------------------


class TestAssignNested:
    def test_simple_dict_key(self):
        container = {}
        assign_nested(container, ["name"], "Alice")
        assert container == {"name": "Alice"}

    def test_nested_dict(self):
        container = {}
        assign_nested(container, ["user", "age"], 30)
        assert container == {"user": {"age": 30}}

    def test_list_index(self):
        container = []
        assign_nested(container, [0], "first")
        assert container == ["first"]

    def test_list_index_extends(self):
        container = []
        assign_nested(container, [2], "third")
        assert container == [None, None, "third"]

    def test_dict_with_list_value(self):
        container = {}
        assign_nested(container, ["items", 0], "x")
        assert container == {"items": ["x"]}

    def test_nested_list_in_dict(self):
        container = {}
        assign_nested(container, ["data", 1], "second")
        assert container["data"] == [None, "second"]

    def test_overwrite_existing(self):
        container = {"key": "old"}
        assign_nested(container, ["key"], "new")
        assert container == {"key": "new"}

    def test_deep_nesting(self):
        container = {}
        assign_nested(container, ["a", "b", "c"], 42)
        assert container["a"]["b"]["c"] == 42


# ---------------------------------------------------------------------------
# parse_url_args
# ---------------------------------------------------------------------------


class TestParseUrlArgs:
    def test_simple_flat(self):
        result = parse_url_args({"name": "Bob"})
        assert result == {"name": "Bob"}

    def test_stringified_json_value(self):
        result = parse_url_args({"filters": '["x","y"]'})
        assert result == {"filters": ["x", "y"]}

    def test_nested_via_bracket(self):
        result = parse_url_args({"user[name]": "Alice"})
        assert result == {"user": {"name": "Alice"}}

    def test_list_via_bracket_index(self):
        result = parse_url_args({"items[0]": "a", "items[1]": "b"})
        assert result == {"items": ["a", "b"]}

    def test_empty_dict(self):
        assert parse_url_args({}) == {}


# ---------------------------------------------------------------------------
# clean_parse_url_args
# ---------------------------------------------------------------------------


class TestCleanParseUrlArgs:
    def test_single_value_list(self):
        result = clean_parse_url_args({"page": ["1"]})
        assert result == {"page": 1}

    def test_multi_value_valid_list(self):
        result = clean_parse_url_args({"tags": ["a", "b", "c"]})
        assert result == {"tags": ["a", "b", "c"]}

    def test_multi_value_invalid_list_flattened(self):
        # When the list contains non-scalar items (dicts), is_valid_value_list is False.
        # Each item is parsed and the results are flattened into a single list.
        result = clean_parse_url_args({"filters": [{"a": 1}, {"b": 2}]})
        # Each dict is not a valid scalar, so the invalid branch runs
        assert result == {"filters": [{"a": 1}, {"b": 2}]}

    def test_non_list_value(self):
        result = clean_parse_url_args({"limit": "50"})
        assert result == {"limit": 50}

    def test_stringified_dict_value(self):
        result = clean_parse_url_args({"meta": '{"k": "v"}'})
        assert result == {"meta": {"k": "v"}}

    def test_empty_params(self):
        assert clean_parse_url_args({}) == {}

    def test_query_key_preserved_as_string(self):
        result = clean_parse_url_args({"query_id": ["42"]})
        assert result == {"query_id": "42"}
