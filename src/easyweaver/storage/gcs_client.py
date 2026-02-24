import io
import json

import polars as pl
import structlog
from google.cloud import storage
from google.oauth2 import service_account

from easyweaver.core.exceptions import StorageError
from easyweaver.settings import settings

logger = structlog.get_logger()

_client: "GCSClient | None" = None


class GCSClient:
    def __init__(
        self,
        bucket_name: str,
        credentials_path: str = "",
        credentials_json: str = "",
    ):
        if not bucket_name:
            raise StorageError("GCP bucket name is not configured (EW_GCP_BUCKET_NAME)")
        self._bucket_name = bucket_name
        if credentials_json:
            info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(info)
            self._storage_client = storage.Client(credentials=credentials)
        elif credentials_path:
            self._storage_client = storage.Client.from_service_account_json(credentials_path)
        else:
            self._storage_client = storage.Client()
        self._bucket = self._storage_client.bucket(bucket_name)
        logger.info("gcs_client_initialized", bucket=bucket_name)

    def upload_parquet(self, path: str, df: pl.DataFrame) -> None:
        """Serialize a Polars DataFrame to Parquet and upload to GCS."""
        try:
            buf = io.BytesIO()
            df.write_parquet(buf)
            buf.seek(0)
            blob = self._bucket.blob(path)
            blob.upload_from_file(buf, content_type="application/octet-stream")
            logger.info("gcs_upload_parquet", path=path, rows=len(df))
        except StorageError:
            raise
        except Exception as exc:
            logger.error("gcs_upload_parquet_failed", path=path, error=str(exc))
            raise StorageError(f"Failed to upload Parquet to GCS: {exc}") from exc

    def download_parquet(self, path: str) -> pl.DataFrame:
        """Download a Parquet blob from GCS and return as a Polars DataFrame."""
        try:
            blob = self._bucket.blob(path)
            buf = io.BytesIO()
            blob.download_to_file(buf)
            buf.seek(0)
            df = pl.read_parquet(buf)
            logger.info("gcs_download_parquet", path=path, rows=len(df))
            return df
        except StorageError:
            raise
        except Exception as exc:
            logger.error("gcs_download_parquet_failed", path=path, error=str(exc))
            raise StorageError(f"Failed to download Parquet from GCS: {exc}") from exc

    def upload_json(self, path: str, data: dict) -> None:
        """Serialize a dict to JSON and upload to GCS."""
        try:
            blob = self._bucket.blob(path)
            blob.upload_from_string(
                json.dumps(data, default=str), content_type="application/json"
            )
            logger.info("gcs_upload_json", path=path)
        except StorageError:
            raise
        except Exception as exc:
            logger.error("gcs_upload_json_failed", path=path, error=str(exc))
            raise StorageError(f"Failed to upload JSON to GCS: {exc}") from exc

    def download_json(self, path: str) -> dict:
        """Download a JSON blob from GCS and deserialize to dict."""
        try:
            blob = self._bucket.blob(path)
            content = blob.download_as_text()
            data = json.loads(content)
            logger.info("gcs_download_json", path=path)
            return data
        except StorageError:
            raise
        except Exception as exc:
            logger.error("gcs_download_json_failed", path=path, error=str(exc))
            raise StorageError(f"Failed to download JSON from GCS: {exc}") from exc


def get_gcs_client() -> GCSClient:
    """Get or create the singleton GCS client."""
    global _client
    if _client is None:
        _client = GCSClient(
            bucket_name=settings.gcs_bucket_name,
            credentials_path=settings.gcs_credentials_path,
            credentials_json=settings.gcs_credentials_json,
        )
    return _client
