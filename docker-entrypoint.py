#!/usr/bin/env python3
"""Docker entrypoint — loads server.env.{env}.json as env vars before starting uvicorn."""
import json
import os
from pathlib import Path


def main():
    config_path = os.environ.get("CONFIG_PATH", "/app/config")
    environment = os.environ.get("EASYWEAVER_ENVIRONMENT", "dev")

    server_env_file = Path(config_path) / "server" / f"server.env.{environment}.json"
    if server_env_file.exists():
        with open(server_env_file) as f:
            defaults = json.load(f)
        for key, value in defaults.items():
            env_key = f"EASYWEAVER_{key}".upper()
            if env_key not in os.environ:
                if isinstance(value, (dict, list)):
                    os.environ[env_key] = json.dumps(value)
                elif isinstance(value, bool):
                    os.environ[env_key] = str(value).lower()
                else:
                    os.environ[env_key] = str(value)

    os.execvp("uvicorn", ["uvicorn", "easyweaver.main:app", "--host", "0.0.0.0", "--port", "8001"])


if __name__ == "__main__":
    main()
