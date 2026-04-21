"""Utilities for handling PEM certificate bundles."""
import os
import re
import shutil
from typing import Optional


def format_pem_bundle(raw_data: str) -> str:
    """Convert raw PEM bundle string into properly formatted PEM file content.

    Environment variables store certificates as a single line with literal \\n
    or spaces. This converts them to proper multi-line PEM format:

        -----BEGIN CERTIFICATE-----
        MIIBxTCCAW... (base64, 64 chars per line)
        -----END CERTIFICATE-----
        -----BEGIN RSA PRIVATE KEY-----
        MIIEvgIBADA... (base64, 64 chars per line)
        -----END RSA PRIVATE KEY-----
    """
    # Replace literal \n with actual newlines
    data = raw_data.replace("\\n", "\n")

    # Extract all PEM blocks (CERTIFICATE, PRIVATE KEY, RSA PRIVATE KEY, etc.)
    pem_pattern = re.compile(
        r"(-----BEGIN [^-]+-----)\s*(.*?)\s*(-----END [^-]+-----)",
        re.DOTALL,
    )

    blocks = []
    for match in pem_pattern.finditer(data):
        begin_line = match.group(1)
        body = match.group(2)
        end_line = match.group(3)

        # Strip all whitespace from body, then re-wrap at 64 chars (PEM standard)
        body_clean = re.sub(r"\s+", "", body)
        wrapped = "\n".join(
            body_clean[i : i + 64] for i in range(0, len(body_clean), 64)
        )

        blocks.append(f"{begin_line}\n{wrapped}\n{end_line}")

    if not blocks:
        # No PEM blocks found — return data with literal \n replaced
        return data

    return "\n".join(blocks) + "\n"


def setup_jira_ssl_bundle(
    jira_config: Optional[dict], config_path: str
) -> Optional[str]:
    """Create PEM file from jira ssl bundle_data. Returns path to PEM file or None.

    Deletes and recreates the bundle directory on every server start.
    Formats raw bundle_data into proper multi-line PEM before writing.
    """
    if not jira_config:
        return None
    ssl = jira_config.get("ssl")
    if not ssl:
        return None
    bundle_data = ssl.get("bundle_data")
    bundle_path = ssl.get("bundle_path")
    bundle_file_name = ssl.get("bundle_file_name")
    if not bundle_data or not bundle_path or not bundle_file_name:
        return None

    # Resolve full directory path relative to config_path (normpath for cross-platform)
    full_dir = os.path.normpath(os.path.join(config_path, bundle_path))

    # Delete and recreate directory every server start
    if os.path.exists(full_dir):
        shutil.rmtree(full_dir)
    os.makedirs(full_dir, exist_ok=True)

    # Format and write PEM file
    pem_path = os.path.normpath(os.path.join(full_dir, bundle_file_name))
    formatted = format_pem_bundle(bundle_data)
    with open(pem_path, "w") as f:
        f.write(formatted)

    return pem_path
