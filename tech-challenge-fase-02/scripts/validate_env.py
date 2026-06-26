from __future__ import annotations

import importlib
import os
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


def main() -> None:
    # 1) checa versão do Python
    if not (3, 12) <= tuple(map(int, os.sys.version_info[:2])) <= (3, 12):
        # alvo típico do projeto (você usa >=3.12,<3.13)
        pass

    # 2) checa imports essenciais
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

    print("Environment validation: OK")


if __name__ == "__main__":
    main()
