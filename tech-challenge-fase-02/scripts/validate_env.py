from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable


def _check_imports(packages: Iterable[str]) -> None:
    """Check if required Python packages can be imported.

    Args:
        packages: Iterable with package names to validate.

    Raises:
        RuntimeError: If one or more packages are missing or cannot be imported.
    """
    missing: list[str] = []

    for package in packages:
        try:
            importlib.import_module(package)
        except Exception:
            missing.append(package)

    if missing:
        raise RuntimeError(f"Missing/unimportable packages: {missing}")


def _check_env_file() -> None:
    """Validate if the .env.example file exists and has required variables.

    Raises:
        RuntimeError: If the .env.example file is missing or does not contain
            required environment variables.
    """
    env_example = Path(".env.example")

    if not env_example.exists():
        raise RuntimeError("Missing .env.example in project root")

    required_vars = ["MLFLOW_TRACKING_URI", "DVC_REMOTE"]
    content = env_example.read_text(encoding="utf-8")

    missing_vars = [var for var in required_vars if f"{var}=" not in content]

    if missing_vars:
        raise RuntimeError(f".env.example missing variables: {missing_vars}")


def main() -> None:
    """Run environment validation checks."""
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
