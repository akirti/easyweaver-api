#!/usr/bin/env python3
"""Run EasyWeaver API locally.

Usage:
    python run.py                  # default: host=0.0.0.0, port=8001, dev environment
    python run.py --port 8003      # custom port
    python run.py --env production # load production config
    python run.py --reload         # auto-reload on code changes (default in dev)
"""
import argparse
import os
import sys
from pathlib import Path


def setup_python_path():
    """Add the src directory to Python path so 'easyweaver' package is importable."""
    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def main():
    parser = argparse.ArgumentParser(description="Run EasyWeaver API locally")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    parser.add_argument("--env", default="dev", help="Environment: dev|staging|production (default: dev)")
    parser.add_argument("--reload", action="store_true", default=None, help="Enable auto-reload")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (default: 1)")
    args = parser.parse_args()

    # Set config environment
    os.environ.setdefault("EASYWEAVER_ENVIRONMENT", args.env)
    os.environ.setdefault("CONFIG_PATH", str(Path(__file__).parent / "config"))

    # Auto-reload in dev by default
    reload_enabled = True if args.env == "dev" else False
    if args.reload:
        reload_enabled = True
    if args.no_reload:
        reload_enabled = False

    print(f"Starting EasyWeaver API")
    print(f"  Environment : {args.env}")
    print(f"  Config path : {os.environ['CONFIG_PATH']}")
    print(f"  Bind        : {args.host}:{args.port}")
    print(f"  Reload      : {reload_enabled}")
    print(f"  Workers     : {args.workers}")
    print()

    import uvicorn
    uvicorn.run(
        "easyweaver.main:app",
        host=args.host,
        port=args.port,
        reload=reload_enabled,
        reload_dirs=[str(Path(__file__).parent / "src")] if reload_enabled else None,
        workers=args.workers if not reload_enabled else 1,
        log_level="debug" if args.env == "dev" else "info",
    )


if __name__ == "__main__":
    setup_python_path()
    main()
