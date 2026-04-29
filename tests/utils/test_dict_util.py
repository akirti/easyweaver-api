"""Tests for easyweaver.utils.dict_util"""
import json
import pytest

from easyweaver.utils.dict_util import DictUtil, DictUtilError, parse_ruby_hash


# ---------------------------------------------------------------------------
# parse_ruby_hash
# ---------------------------------------------------------------------------


class TestParseRubyHash:
    def test_basic_ruby_hash(self):
        raw = '{:key=>"value"}'
        result = parse_ruby_hash(raw)
        assert result == {"key": "value"}

    def test_multiple_pairs(self):
        raw = '{:name=>"Alice", :age=>"30"}'
        result = parse_ruby_hash(raw)
        assert result == {"name": "Alice", "age": "30"}

    def test_symbol_without_colon_prefix(self):
        raw = '{name=>"Bob"}'
        result = parse_ruby_hash(raw)
        assert result == {"name": "Bob"}

    def test_not_a_ruby_hash_returns_none(self):
        assert parse_ruby_hash("hello world") is None

    def test_no_arrow_returns_none(self):
        assert parse_ruby_hash('{"key": "value"}') is None

    def test_no_braces_returns_none(self):
        assert parse_ruby_hash(':key=>"val"') is None

    def test_empty_hash_body_returns_none(self):
        # no pairs matched → result dict is empty → returns None
        assert parse_ruby_hash("{}") is None

    def test_escaped_quote_in_value(self):
        raw = r'{:msg=>"say \"hi\""}'
        result = parse_ruby_hash(raw)
        assert result == {"msg": 'say "hi"'}

    def test_backslash_escape(self):
        raw = r'{:path=>"C:\\Users"}'
        result = parse_ruby_hash(raw)
        assert result == {"path": "C:\\Users"}

    def test_newline_escape(self):
        raw = '{:text=>"line1\\nline2"}'
        result = parse_ruby_hash(raw)
        assert result["text"] == "line1\nline2"

    def test_pem_value_passthrough(self):
        raw = '{:cert=>"-----BEGIN CERTIFICATE-----\\nMIIBxTC\\n-----END CERTIFICATE-----"}'
        result = parse_ruby_hash(raw)
        # PEM values must NOT have escape sequences replaced
        assert result is not None
        assert "-----BEGIN CERTIFICATE-----" in result["cert"]

    def test_key_with_dots_and_dashes(self):
        # Pattern supports dots and dashes in bare key names (no surrounding quotes)
        raw = '{:my-key.val=>"x"}'
        result = parse_ruby_hash(raw)
        # the leading : is stripped, key becomes "my-key.val"
        assert result is not None
        assert result.get("my-key.val") == "x"

    def test_whitespace_around_hash(self):
        raw = '  {:key=>"value"}  '
        result = parse_ruby_hash(raw)
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# DictUtil.deep_get
# ---------------------------------------------------------------------------


class TestDictUtilDeepGet:
    def setup_method(self):
        self.util = DictUtil()

    def test_simple_key(self):
        d = {"a": 1}
        assert self.util.deep_get(d, "a") == 1

    def test_nested_key(self):
        d = {"a": {"b": {"c": 42}}}
        assert self.util.deep_get(d, "a.b.c") == 42

    def test_missing_key_returns_default(self):
        d = {"a": 1}
        assert self.util.deep_get(d, "a.b") is None

    def test_missing_key_custom_default(self):
        d = {"a": 1}
        assert self.util.deep_get(d, "x", default="fallback") == "fallback"

    def test_intermediate_non_dict_returns_default(self):
        d = {"a": "string"}
        assert self.util.deep_get(d, "a.b") is None

    def test_top_level_key(self):
        d = {"flat": "value"}
        assert self.util.deep_get(d, "flat") == "value"

    def test_empty_dict(self):
        assert self.util.deep_get({}, "any.path") is None


# ---------------------------------------------------------------------------
# DictUtil.get_deep_nested_value
# ---------------------------------------------------------------------------


class TestDictUtilGetDeepNestedValue:
    def setup_method(self):
        self.util = DictUtil()

    def test_simple_lookup(self):
        d = {"x": {"y": "hello"}}
        assert self.util.get_deep_nested_value(d, "x.y") == "hello"

    def test_deep_nested(self):
        d = {"level1": {"level2": {"key": "deep_value"}}}
        assert self.util.get_deep_nested_value(d, "level1.level2.key") == "deep_value"

    def test_missing_key_returns_none(self):
        d = {"a": {"b": 1}}
        assert self.util.get_deep_nested_value(d, "a.c") is None

    def test_list_of_dicts(self):
        d = {"items": [{"name": "Alice"}, {"name": "Bob"}]}
        result = self.util.get_deep_nested_value(d, "items.name")
        assert result == ["Alice", "Bob"]

    def test_attribute_error_returns_none(self):
        d = {"a": "not_a_dict"}
        # Accessing .get on a string triggers AttributeError → returns None
        result = self.util.get_deep_nested_value(d, "a.b")
        assert result is None

    def test_missing_top_level_key_returns_none(self):
        # get_deep_nested_value reduces through dict.get() — missing keys yield None
        d = {}
        result = self.util.get_deep_nested_value(d, "missing")
        assert result is None


# ---------------------------------------------------------------------------
# DictUtil.merge_dicts
# ---------------------------------------------------------------------------


class TestDictUtilMergeDicts:
    def setup_method(self):
        self.util = DictUtil()

    def test_adds_missing_keys(self):
        base = {"a": 1}
        extra = {"b": 2}
        self.util.merge_dicts(base, extra)
        assert base == {"a": 1, "b": 2}

    def test_overwrites_scalar_values(self):
        base = {"a": 1}
        extra = {"a": 99}
        self.util.merge_dicts(base, extra)
        assert base["a"] == 99

    def test_recursive_dict_merge(self):
        base = {"nested": {"x": 1}}
        extra = {"nested": {"y": 2}}
        self.util.merge_dicts(base, extra)
        assert base == {"nested": {"x": 1, "y": 2}}

    def test_list_extension_no_duplicates(self):
        base = {"tags": ["a", "b"]}
        extra = {"tags": ["b", "c"]}
        self.util.merge_dicts(base, extra)
        assert sorted(base["tags"]) == ["a", "b", "c"]

    def test_list_extension_adds_only_new(self):
        base = {"items": [1, 2]}
        extra = {"items": [2, 3, 4]}
        self.util.merge_dicts(base, extra)
        assert base["items"] == [1, 2, 3, 4]

    def test_does_not_add_existing_list_items(self):
        base = {"nums": [10, 20]}
        extra = {"nums": [10]}
        self.util.merge_dicts(base, extra)
        assert base["nums"] == [10, 20]

    def test_empty_setting_dict(self):
        base = {"a": 1}
        self.util.merge_dicts(base, {})
        assert base == {"a": 1}

    def test_empty_config_dict(self):
        base = {}
        extra = {"new_key": "new_val"}
        self.util.merge_dicts(base, extra)
        assert base == {"new_key": "new_val"}

    def test_deeply_nested_recursive(self):
        base = {"a": {"b": {"c": 1}}}
        extra = {"a": {"b": {"d": 2}}}
        self.util.merge_dicts(base, extra)
        assert base["a"]["b"] == {"c": 1, "d": 2}

    def test_mixed_types_overwrite(self):
        # base has dict, extra has scalar — scalar wins
        base = {"a": {"b": 1}}
        extra = {"a": "string_overwrite"}
        self.util.merge_dicts(base, extra)
        assert base["a"] == "string_overwrite"


# ---------------------------------------------------------------------------
# DictUtilError
# ---------------------------------------------------------------------------


class TestDictUtilError:
    def test_is_exception(self):
        err = DictUtilError("something went wrong")
        assert isinstance(err, Exception)
        assert str(err) == "something went wrong"

    def test_can_be_raised(self):
        with pytest.raises(DictUtilError, match="test error"):
            raise DictUtilError("test error")
