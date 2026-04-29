"""Tests for easyweaver.core.security"""
import importlib
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_security_module(fernet_key: str):
    """Re-import core.security with a fresh module state and the given fernet_key."""
    import easyweaver.core.security as sec_mod

    # Reset the cached _fernet so _get_fernet() re-initialises
    sec_mod._fernet = None

    with patch.object(sec_mod.settings, "fernet_key", fernet_key):
        yield sec_mod


# ---------------------------------------------------------------------------
# _get_fernet
# ---------------------------------------------------------------------------


class TestGetFernet:
    def setup_method(self):
        import easyweaver.core.security as sec
        sec._fernet = None  # always start fresh

    def test_raises_runtime_error_for_default_key(self):
        import easyweaver.core.security as sec
        default_key = "change-me-generate-with-cryptography-fernet"
        with patch.object(sec.settings, "fernet_key", default_key):
            with pytest.raises(RuntimeError, match="EW_FERNET_KEY not set"):
                sec._get_fernet()

    def test_returns_fernet_instance_for_valid_key(self):
        import easyweaver.core.security as sec
        key = Fernet.generate_key().decode()
        with patch.object(sec.settings, "fernet_key", key):
            fernet = sec._get_fernet()
        assert isinstance(fernet, Fernet)

    def test_caches_fernet_instance(self):
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key().decode()
        with patch.object(sec.settings, "fernet_key", key):
            fernet1 = sec._get_fernet()
            fernet2 = sec._get_fernet()
        assert fernet1 is fernet2

    def test_accepts_bytes_key(self):
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key()  # bytes
        with patch.object(sec.settings, "fernet_key", key):
            fernet = sec._get_fernet()
        assert isinstance(fernet, Fernet)


# ---------------------------------------------------------------------------
# encrypt_credentials / decrypt_credentials
# ---------------------------------------------------------------------------


class TestEncryptDecryptCredentials:
    def setup_method(self):
        import easyweaver.core.security as sec
        sec._fernet = None

    def _patch_key(self, key):
        import easyweaver.core.security as sec
        return patch.object(sec.settings, "fernet_key", key)

    def test_roundtrip(self):
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key().decode()
        with self._patch_key(key):
            plaintext = "super-secret-password"
            ciphertext = sec.encrypt_credentials(plaintext)
            recovered = sec.decrypt_credentials(ciphertext)
        assert recovered == plaintext

    def test_ciphertext_differs_from_plaintext(self):
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key().decode()
        with self._patch_key(key):
            ciphertext = sec.encrypt_credentials("password123")
        assert ciphertext != "password123"

    def test_empty_string_roundtrip(self):
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key().decode()
        with self._patch_key(key):
            ciphertext = sec.encrypt_credentials("")
            recovered = sec.decrypt_credentials(ciphertext)
        assert recovered == ""

    def test_unicode_roundtrip(self):
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key().decode()
        with self._patch_key(key):
            plaintext = "päs\u00dfwort"  # German password with special chars
            ciphertext = sec.encrypt_credentials(plaintext)
            recovered = sec.decrypt_credentials(ciphertext)
        assert recovered == plaintext

    def test_encrypt_returns_string(self):
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key().decode()
        with self._patch_key(key):
            result = sec.encrypt_credentials("test")
        assert isinstance(result, str)

    def test_decrypt_returns_string(self):
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key().decode()
        with self._patch_key(key):
            ct = sec.encrypt_credentials("test")
            result = sec.decrypt_credentials(ct)
        assert isinstance(result, str)

    def test_different_keys_cannot_decrypt(self):
        import easyweaver.core.security as sec
        from cryptography.fernet import InvalidToken

        key1 = Fernet.generate_key().decode()
        key2 = Fernet.generate_key().decode()

        sec._fernet = None
        with patch.object(sec.settings, "fernet_key", key1):
            ciphertext = sec.encrypt_credentials("data")

        sec._fernet = None
        with patch.object(sec.settings, "fernet_key", key2):
            with pytest.raises(Exception):  # InvalidToken or similar
                sec.decrypt_credentials(ciphertext)

    def test_encryption_is_non_deterministic(self):
        """Fernet uses random IVs — same plaintext encrypts to different ciphertext."""
        import easyweaver.core.security as sec
        sec._fernet = None
        key = Fernet.generate_key().decode()
        with patch.object(sec.settings, "fernet_key", key):
            ct1 = sec.encrypt_credentials("same")
            ct2 = sec.encrypt_credentials("same")
        assert ct1 != ct2
