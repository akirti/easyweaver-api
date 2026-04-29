"""Tests for easyweaver.utils.certificate_util"""
import os
import textwrap
from unittest.mock import MagicMock, mock_open, patch

import pytest

from easyweaver.utils.certificate_util import format_pem_bundle, setup_jira_ssl_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CERT = (
    "-----BEGIN CERTIFICATE-----\n"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345abcdefghijklmnopqrstuvwxyz012345\n"
    "-----END CERTIFICATE-----\n"
)

_FAKE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345abcdefghijklmnopqrstuvwxyz012345\n"
    "-----END RSA PRIVATE KEY-----\n"
)


# ---------------------------------------------------------------------------
# format_pem_bundle
# ---------------------------------------------------------------------------


class TestFormatPemBundle:
    def test_literal_newline_replaced(self):
        raw = "-----BEGIN CERTIFICATE-----\\nABCD\\n-----END CERTIFICATE-----"
        result = format_pem_bundle(raw)
        assert "\\n" not in result
        assert "\n" in result

    def test_single_cert_block_formatted(self):
        raw = "-----BEGIN CERTIFICATE-----\\nABCDEFGH\\n-----END CERTIFICATE-----"
        result = format_pem_bundle(raw)
        assert result.startswith("-----BEGIN CERTIFICATE-----\n")
        assert result.strip().endswith("-----END CERTIFICATE-----")

    def test_body_wrapped_at_64_chars(self):
        # Build a body that is 128 chars long — should be split into 2 lines of 64
        body_raw = "A" * 128
        raw = f"-----BEGIN CERTIFICATE-----\\n{body_raw}\\n-----END CERTIFICATE-----"
        result = format_pem_bundle(raw)
        lines = result.strip().split("\n")
        # lines: [BEGIN, line1(64), line2(64), END]
        body_lines = lines[1:-1]
        assert all(len(line) <= 64 for line in body_lines)
        assert len(body_lines) == 2

    def test_multiple_pem_blocks(self):
        raw = (
            "-----BEGIN CERTIFICATE-----\\nABCD\\n-----END CERTIFICATE-----"
            "\\n-----BEGIN RSA PRIVATE KEY-----\\nEFGH\\n-----END RSA PRIVATE KEY-----"
        )
        result = format_pem_bundle(raw)
        assert "-----BEGIN CERTIFICATE-----" in result
        assert "-----BEGIN RSA PRIVATE KEY-----" in result

    def test_no_pem_blocks_returns_data_with_newlines(self):
        raw = "not a pem file at all"
        result = format_pem_bundle(raw)
        assert result == "not a pem file at all"

    def test_whitespace_stripped_from_body(self):
        # Body with spaces — they should be stripped before re-wrapping
        raw = "-----BEGIN CERTIFICATE-----\\n  AB CD EF  \\n-----END CERTIFICATE-----"
        result = format_pem_bundle(raw)
        lines = result.strip().split("\n")
        body_lines = lines[1:-1]
        for line in body_lines:
            assert " " not in line

    def test_output_ends_with_newline(self):
        raw = "-----BEGIN CERTIFICATE-----\\nABCD\\n-----END CERTIFICATE-----"
        result = format_pem_bundle(raw)
        assert result.endswith("\n")

    def test_already_has_real_newlines(self):
        # real newlines (not escaped) also work
        raw = "-----BEGIN CERTIFICATE-----\nABCD\n-----END CERTIFICATE-----"
        result = format_pem_bundle(raw)
        assert "-----BEGIN CERTIFICATE-----" in result
        assert "-----END CERTIFICATE-----" in result

    def test_empty_string(self):
        result = format_pem_bundle("")
        assert result == ""


# ---------------------------------------------------------------------------
# setup_jira_ssl_bundle
# ---------------------------------------------------------------------------


class TestSetupJiraSslBundle:
    def test_none_jira_config_returns_none(self):
        assert setup_jira_ssl_bundle(None, "/some/path") is None

    def test_empty_jira_config_returns_none(self):
        assert setup_jira_ssl_bundle({}, "/some/path") is None

    def test_no_ssl_key_returns_none(self):
        assert setup_jira_ssl_bundle({"host": "jira.example.com"}, "/path") is None

    def test_empty_ssl_returns_none(self):
        assert setup_jira_ssl_bundle({"ssl": {}}, "/path") is None

    def test_missing_bundle_data_returns_none(self):
        config = {"ssl": {"bundle_path": "certs", "bundle_file_name": "bundle.pem"}}
        assert setup_jira_ssl_bundle(config, "/path") is None

    def test_missing_bundle_path_returns_none(self):
        config = {"ssl": {"bundle_data": "data", "bundle_file_name": "bundle.pem"}}
        assert setup_jira_ssl_bundle(config, "/path") is None

    def test_missing_bundle_file_name_returns_none(self):
        config = {"ssl": {"bundle_data": "data", "bundle_path": "certs"}}
        assert setup_jira_ssl_bundle(config, "/path") is None

    def test_creates_directory_and_writes_file(self, tmp_path):
        config = {
            "ssl": {
                "bundle_data": "-----BEGIN CERTIFICATE-----\\nABCD\\n-----END CERTIFICATE-----",
                "bundle_path": "certs",
                "bundle_file_name": "bundle.pem",
            }
        }
        pem_path = setup_jira_ssl_bundle(config, str(tmp_path))
        assert pem_path is not None
        assert os.path.isfile(pem_path)
        content = open(pem_path).read()
        assert "-----BEGIN CERTIFICATE-----" in content

    def test_recreates_directory_on_second_call(self, tmp_path):
        config = {
            "ssl": {
                "bundle_data": "-----BEGIN CERTIFICATE-----\\nABCD\\n-----END CERTIFICATE-----",
                "bundle_path": "ssl_certs",
                "bundle_file_name": "bundle.pem",
            }
        }
        # First call
        path1 = setup_jira_ssl_bundle(config, str(tmp_path))
        # Write an extra file into the directory
        extra = os.path.join(os.path.dirname(path1), "extra.txt")
        with open(extra, "w") as f:
            f.write("should be removed")
        # Second call must recreate dir (rmtree + makedirs)
        path2 = setup_jira_ssl_bundle(config, str(tmp_path))
        assert not os.path.exists(extra), "directory should have been recreated"
        assert os.path.isfile(path2)

    def test_returned_path_uses_config_path(self, tmp_path):
        config = {
            "ssl": {
                "bundle_data": "-----BEGIN CERTIFICATE-----\\nABCD\\n-----END CERTIFICATE-----",
                "bundle_path": "subdir",
                "bundle_file_name": "my.pem",
            }
        }
        pem_path = setup_jira_ssl_bundle(config, str(tmp_path))
        assert pem_path.endswith("my.pem")
        assert "subdir" in pem_path
