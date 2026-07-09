from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI

app = FastAPI(
    title="Tech Challenge Fase 02 - Recommender API",
    description="API pública para validação do deploy em nuvem.",
    version="1.0.0",
)

METRICS_FILE = Path("reports/torch_metrics.json")
MODEL_FILE = Path("models/recommender_net.pt")


def load_metrics() -> dict[str, Any]:
    """Carrega as métricas finais do modelo, caso o arquivo esteja disponível."""
    if not METRICS_FILE.exists():
        return {
            "status": "metrics_not_found",
            "message": "Arquivo reports/torch_metrics.json não encontrado.",
        }

    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


@app.get("/")
def root() -> dict[str, str]:
    """Retorna uma mensagem simples para validar se a API está online."""
    return {
        "project": "Tech Challenge Fase 02",
        "message": "Recommender API is running",
        "status": "online",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint de health check usado para validar o container."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Retorna informações gerais do modelo e métricas finais."""
    return {
        "project": "tech-challenge-fase-02",
        "model": "RecommenderNet",
        "model_registry": {
            "registered_model_name": "RecommenderNet",
            "alias": "candidate",
        },
        "model_file_available": MODEL_FILE.exists(),
        "metrics": load_metrics(),
        "note": (
            "O modelo treinado é controlado localmente/DVC. "
            "Esta API pública expõe informações e métricas do projeto."
        ),
    }
