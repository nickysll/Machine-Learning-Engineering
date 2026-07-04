from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable


def _check_imports(packages: Iterable[str]) -> None:
    missing: list[str] = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)

    if missing:
        raise RuntimeError(f"Missing/unimportable packages: {missing}")


def _check_env_file() -> None:
    env_example = Path(".env.example")
    if not env_example.exists():
        raise RuntimeError("Missing .env.example in project root")

    required_vars = ["MLFLOW_TRACKING_URI", "DVC_REMOTE"]
    content = env_example.read_text(encoding="utf-8")

    missing_vars = [v for v in required_vars if v + "=" not in content]
    if missing_vars:
        raise RuntimeError(f".env.example missing variables: {missing_vars}")


def main() -> None:
    _check_imports(
        [
            "numpy",
            "pandas",
            "sklearn",
            "torch",
            "mlflow",
            "dvc",
            "requests",
            "yaml",
        ]
    )
    _check_env_file()

    print("Environment validation: OK")


if __name__ == "__main__":
    main()
