"""Tests for easyweaver.storage.gcs_client — GCSClient with mocked GCP."""
import io
import json
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from easyweaver.core.exceptions import StorageError
from easyweaver.storage.gcs_client import GCSClient, get_gcs_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(rows: int = 5) -> pl.DataFrame:
    return pl.DataFrame({"id": list(range(rows)), "name": [f"item_{i}" for i in range(rows)]})


def _make_parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def _make_client(
    bucket_name: str = "test-bucket",
    credentials_json: str = "",
    credentials_path: str = "",
) -> GCSClient:
    """Build a GCSClient with mocked google.cloud.storage."""
    mock_storage_client = MagicMock()
    mock_bucket = MagicMock()
    mock_storage_client.bucket = MagicMock(return_value=mock_bucket)

    with patch("easyweaver.storage.gcs_client.storage") as mock_storage_module:
        mock_storage_module.Client.return_value = mock_storage_client
        mock_storage_module.Client.from_service_account_json.return_value = mock_storage_client
        client = GCSClient(
            bucket_name=bucket_name,
            credentials_json=credentials_json,
            credentials_path=credentials_path,
        )
    # Directly attach mocks so tests can assert on them
    client._storage_client = mock_storage_client
    client._bucket = mock_bucket
    return client


# ---------------------------------------------------------------------------
# GCSClient.__init__
# ---------------------------------------------------------------------------


class TestGCSClientInit:
    def test_raises_storage_error_when_no_bucket(self):
        with pytest.raises(StorageError, match="bucket name"):
            with patch("easyweaver.storage.gcs_client.storage"):
                GCSClient(bucket_name="")

    def test_uses_default_credentials_when_none_provided(self):
        mock_storage_client = MagicMock()
        mock_storage_client.bucket = MagicMock(return_value=MagicMock())
        with patch("easyweaver.storage.gcs_client.storage") as mock_storage_mod:
            mock_storage_mod.Client.return_value = mock_storage_client
            client = GCSClient(bucket_name="my-bucket")
        assert client._bucket_name == "my-bucket"

    def test_uses_credentials_path_when_provided(self):
        mock_storage_client = MagicMock()
        mock_storage_client.bucket = MagicMock(return_value=MagicMock())
        with patch("easyweaver.storage.gcs_client.storage") as mock_storage_mod:
            mock_storage_mod.Client.from_service_account_json.return_value = mock_storage_client
            client = GCSClient(
                bucket_name="my-bucket", credentials_path="/path/to/creds.json"
            )
        mock_storage_mod.Client.from_service_account_json.assert_called_once_with(
            "/path/to/creds.json"
        )

    def test_uses_credentials_json_when_provided(self):
        creds_info = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        mock_credentials = MagicMock()
        mock_storage_client = MagicMock()
        mock_storage_client.bucket = MagicMock(return_value=MagicMock())

        with patch("easyweaver.storage.gcs_client.storage") as mock_storage_mod:
            with patch(
                "easyweaver.storage.gcs_client.service_account"
            ) as mock_sa:
                mock_sa.Credentials.from_service_account_info.return_value = mock_credentials
                mock_storage_mod.Client.return_value = mock_storage_client
                client = GCSClient(
                    bucket_name="my-bucket",
                    credentials_json=json.dumps(creds_info),
                )
        mock_sa.Credentials.from_service_account_info.assert_called_once()


# ---------------------------------------------------------------------------
# upload_parquet
# ---------------------------------------------------------------------------


class TestUploadParquet:
    def test_uploads_parquet_successfully(self):
        client = _make_client()
        mock_blob = MagicMock()
        client._bucket.blob = MagicMock(return_value=mock_blob)

        df = _make_df()
        client.upload_parquet("results/test.parquet", df)

        client._bucket.blob.assert_called_once_with("results/test.parquet")
        mock_blob.upload_from_file.assert_called_once()

    def test_upload_parquet_passes_correct_content_type(self):
        client = _make_client()
        mock_blob = MagicMock()
        client._bucket.blob = MagicMock(return_value=mock_blob)

        df = _make_df()
        client.upload_parquet("data.parquet", df)

        call_kwargs = mock_blob.upload_from_file.call_args
        assert call_kwargs[1].get("content_type") == "application/octet-stream"

    def test_upload_parquet_raises_storage_error_on_failure(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.upload_from_file = MagicMock(side_effect=Exception("GCS error"))
        client._bucket.blob = MagicMock(return_value=mock_blob)

        df = _make_df()
        with pytest.raises(StorageError, match="Failed to upload Parquet"):
            client.upload_parquet("data.parquet", df)

    def test_upload_parquet_re_raises_storage_error(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.upload_from_file = MagicMock(side_effect=StorageError("already a storage error"))
        client._bucket.blob = MagicMock(return_value=mock_blob)

        df = _make_df()
        with pytest.raises(StorageError, match="already a storage error"):
            client.upload_parquet("data.parquet", df)

    def test_upload_parquet_serializes_dataframe(self):
        captured = {}
        client = _make_client()
        mock_blob = MagicMock()

        def capture_upload(buf, **kwargs):
            captured["bytes"] = buf.read()

        mock_blob.upload_from_file = MagicMock(side_effect=capture_upload)
        client._bucket.blob = MagicMock(return_value=mock_blob)

        df = _make_df(3)
        client.upload_parquet("data.parquet", df)

        # Verify valid parquet bytes
        recovered = pl.read_parquet(io.BytesIO(captured["bytes"]))
        assert recovered.shape == df.shape


# ---------------------------------------------------------------------------
# download_parquet
# ---------------------------------------------------------------------------


class TestDownloadParquet:
    def test_downloads_parquet_successfully(self):
        client = _make_client()
        df = _make_df(4)
        parquet_bytes = _make_parquet_bytes(df)

        mock_blob = MagicMock()

        def fake_download(buf):
            buf.write(parquet_bytes)

        mock_blob.download_to_file = MagicMock(side_effect=fake_download)
        client._bucket.blob = MagicMock(return_value=mock_blob)

        result = client.download_parquet("results/data.parquet")
        assert isinstance(result, pl.DataFrame)
        assert result.shape == df.shape

    def test_download_parquet_raises_storage_error_on_failure(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.download_to_file = MagicMock(side_effect=Exception("not found"))
        client._bucket.blob = MagicMock(return_value=mock_blob)

        with pytest.raises(StorageError, match="Failed to download Parquet"):
            client.download_parquet("missing.parquet")

    def test_download_parquet_re_raises_storage_error(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.download_to_file = MagicMock(
            side_effect=StorageError("direct storage error")
        )
        client._bucket.blob = MagicMock(return_value=mock_blob)

        with pytest.raises(StorageError, match="direct storage error"):
            client.download_parquet("data.parquet")

    def test_download_parquet_uses_correct_path(self):
        client = _make_client()
        df = _make_df()
        parquet_bytes = _make_parquet_bytes(df)
        mock_blob = MagicMock()
        mock_blob.download_to_file = MagicMock(
            side_effect=lambda buf: buf.write(parquet_bytes)
        )
        client._bucket.blob = MagicMock(return_value=mock_blob)

        client.download_parquet("specific/path/data.parquet")
        client._bucket.blob.assert_called_once_with("specific/path/data.parquet")


# ---------------------------------------------------------------------------
# upload_json
# ---------------------------------------------------------------------------


class TestUploadJson:
    def test_uploads_json_successfully(self):
        client = _make_client()
        mock_blob = MagicMock()
        client._bucket.blob = MagicMock(return_value=mock_blob)

        data = {"key": "value", "count": 42}
        client.upload_json("results/meta.json", data)

        mock_blob.upload_from_string.assert_called_once()
        call_args = mock_blob.upload_from_string.call_args
        assert call_args[1].get("content_type") == "application/json"

    def test_upload_json_serializes_to_json_string(self):
        client = _make_client()
        captured = {}
        mock_blob = MagicMock()
        mock_blob.upload_from_string = MagicMock(
            side_effect=lambda content, **kw: captured.update({"content": content})
        )
        client._bucket.blob = MagicMock(return_value=mock_blob)

        data = {"name": "test", "value": 123}
        client.upload_json("meta.json", data)

        parsed = json.loads(captured["content"])
        assert parsed == data

    def test_upload_json_raises_storage_error_on_failure(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.upload_from_string = MagicMock(side_effect=Exception("upload failed"))
        client._bucket.blob = MagicMock(return_value=mock_blob)

        with pytest.raises(StorageError, match="Failed to upload JSON"):
            client.upload_json("meta.json", {"x": 1})

    def test_upload_json_re_raises_storage_error(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.upload_from_string = MagicMock(
            side_effect=StorageError("direct json error")
        )
        client._bucket.blob = MagicMock(return_value=mock_blob)

        with pytest.raises(StorageError, match="direct json error"):
            client.upload_json("meta.json", {})

    def test_upload_json_handles_non_serializable_with_default_str(self):
        """upload_json uses default=str, so non-serializable objects become strings."""
        from datetime import datetime
        client = _make_client()
        mock_blob = MagicMock()
        captured = {}
        mock_blob.upload_from_string = MagicMock(
            side_effect=lambda content, **kw: captured.update({"content": content})
        )
        client._bucket.blob = MagicMock(return_value=mock_blob)

        now = datetime.now()
        client.upload_json("meta.json", {"timestamp": now})
        parsed = json.loads(captured["content"])
        assert isinstance(parsed["timestamp"], str)


# ---------------------------------------------------------------------------
# download_json
# ---------------------------------------------------------------------------


class TestDownloadJson:
    def test_downloads_json_successfully(self):
        client = _make_client()
        expected = {"result": "success", "count": 5}
        mock_blob = MagicMock()
        mock_blob.download_as_text = MagicMock(return_value=json.dumps(expected))
        client._bucket.blob = MagicMock(return_value=mock_blob)

        result = client.download_json("results/meta.json")
        assert result == expected

    def test_download_json_raises_storage_error_on_failure(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.download_as_text = MagicMock(side_effect=Exception("blob not found"))
        client._bucket.blob = MagicMock(return_value=mock_blob)

        with pytest.raises(StorageError, match="Failed to download JSON"):
            client.download_json("missing.json")

    def test_download_json_re_raises_storage_error(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.download_as_text = MagicMock(
            side_effect=StorageError("direct download error")
        )
        client._bucket.blob = MagicMock(return_value=mock_blob)

        with pytest.raises(StorageError, match="direct download error"):
            client.download_json("data.json")

    def test_download_json_uses_correct_path(self):
        client = _make_client()
        mock_blob = MagicMock()
        mock_blob.download_as_text = MagicMock(return_value='{"a":1}')
        client._bucket.blob = MagicMock(return_value=mock_blob)

        client.download_json("specific/meta.json")
        client._bucket.blob.assert_called_once_with("specific/meta.json")


# ---------------------------------------------------------------------------
# get_gcs_client — singleton
# ---------------------------------------------------------------------------


class TestGetGcsClient:
    def setup_method(self):
        # Reset singleton
        import easyweaver.storage.gcs_client as gcs_module
        gcs_module._client = None

    def test_creates_client_on_first_call(self):
        mock_storage_client = MagicMock()
        mock_storage_client.bucket = MagicMock(return_value=MagicMock())

        with patch("easyweaver.storage.gcs_client.storage") as mock_storage_mod:
            mock_storage_mod.Client.return_value = mock_storage_client
            with patch("easyweaver.storage.gcs_client.settings") as mock_settings:
                mock_settings.gcs_bucket_name = "my-bucket"
                mock_settings.gcs_credentials_path = ""
                mock_settings.gcs_credentials_json = ""
                client = get_gcs_client()

        assert isinstance(client, GCSClient)

    def test_returns_same_instance_on_second_call(self):
        mock_storage_client = MagicMock()
        mock_storage_client.bucket = MagicMock(return_value=MagicMock())

        with patch("easyweaver.storage.gcs_client.storage") as mock_storage_mod:
            mock_storage_mod.Client.return_value = mock_storage_client
            with patch("easyweaver.storage.gcs_client.settings") as mock_settings:
                mock_settings.gcs_bucket_name = "my-bucket"
                mock_settings.gcs_credentials_path = ""
                mock_settings.gcs_credentials_json = ""
                client1 = get_gcs_client()
                client2 = get_gcs_client()

        assert client1 is client2

    def test_raises_storage_error_when_no_bucket_configured(self):
        import easyweaver.storage.gcs_client as gcs_module
        gcs_module._client = None

        with patch("easyweaver.storage.gcs_client.storage"):
            with patch("easyweaver.storage.gcs_client.settings") as mock_settings:
                mock_settings.gcs_bucket_name = ""
                mock_settings.gcs_credentials_path = ""
                mock_settings.gcs_credentials_json = ""
                with pytest.raises(StorageError):
                    get_gcs_client()
