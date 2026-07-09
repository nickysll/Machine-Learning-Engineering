from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import compute_classification_metrics
from src.models.factory import build_model

TRAIN_FILE = Path("data/features/train.csv")
TEST_FILE = Path("data/features/test.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_FILE = MODELS_DIR / "recommender_net.pt"
REPORT_FILE = REPORTS_DIR / "torch_metrics.json"

TARGET_COLUMN = "target"
EXPERIMENT_NAME = "tech-challenge-recommender"

REGISTERED_MODEL_NAME = "RecommenderNet"
MODEL_ALIAS = "candidate"


class InteractionDataset(Dataset):
    """Dataset PyTorch para interações entre usuários e itens.

    A classe recebe uma base já processada, com identificadores numéricos
    para usuários e itens, e transforma esses campos em tensores para uso
    no treinamento da rede neural.

    Expected columns:
        - user_id_idx: identificador interno do usuário.
        - item_id_idx: identificador interno do item.
        - target: classe binária da interação.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Inicializa os tensores usados pelo DataLoader.

        Args:
            data: DataFrame contendo usuários, itens e target binário.
        """
        self.user_ids = torch.tensor(data["user_id_idx"].values, dtype=torch.long)
        self.item_ids = torch.tensor(data["item_id_idx"].values, dtype=torch.long)
        self.targets = torch.tensor(data[TARGET_COLUMN].values, dtype=torch.float32)

    def __len__(self) -> int:
        """Retorna a quantidade de interações disponíveis no dataset."""
        return len(self.targets)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna uma amostra individual do dataset.

        Args:
            index: Posição da interação no dataset.

        Returns:
            Tupla com identificador do usuário, identificador do item e target.
        """
        return self.user_ids[index], self.item_ids[index], self.targets[index]


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando do treinamento.

    Returns:
        Namespace com os hiperparâmetros e configurações do treinamento.
    """
    parser = argparse.ArgumentParser(description="Train PyTorch recommender model.")
    parser.add_argument("--run-name", type=str, default="recommender_net")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Define seeds para melhorar a reprodutibilidade do treinamento.

    Args:
        seed: Valor usado para controlar aleatoriedade em Python, NumPy e PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_name: str) -> torch.device:
    """Define o dispositivo usado no treinamento.

    Args:
        device_name: Nome do dispositivo escolhido. Pode ser "cpu", "cuda" ou "auto".

    Returns:
        Dispositivo PyTorch que será usado no treinamento.

    Raises:
        RuntimeError: Caso CUDA seja solicitado explicitamente, mas não esteja
            disponível no ambiente.
    """
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return torch.device("cuda")

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) >= (7, 5):
            return torch.device("cuda")

    return torch.device("cpu")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Carrega um dataset em formato CSV.

    Args:
        file_path: Caminho do arquivo CSV.

    Returns:
        DataFrame com os dados carregados.
    """
    return pd.read_csv(file_path)


def split_train_validation(
    data: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa a base de treino em treino e validação.

    A separação mantém a proporção da variável target por meio de stratify,
    reduzindo o risco de criar uma validação muito diferente do treino.

    Args:
        data: DataFrame de treino.
        seed: Seed usada na separação.

    Returns:
        Tupla contendo DataFrame de treino e DataFrame de validação.
    """
    train_data, validation_data = train_test_split(
        data,
        test_size=0.1,
        random_state=seed,
        stratify=data[TARGET_COLUMN],
    )

    return train_data, validation_data


def create_dataloader(
    data: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Cria um DataLoader PyTorch a partir de um DataFrame.

    Args:
        data: DataFrame com interações.
        batch_size: Tamanho do lote usado no treinamento ou avaliação.
        shuffle: Indica se os dados devem ser embaralhados.

    Returns:
        DataLoader pronto para ser usado pelo modelo.
    """
    dataset = InteractionDataset(data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def get_model_sizes(dataframes: list[pd.DataFrame]) -> tuple[int, int]:
    """Calcula a quantidade total de usuários e itens.

    Essa informação é necessária para inicializar as camadas de embedding
    do modelo, já que elas precisam saber quantos usuários e itens existem.

    Args:
        dataframes: Lista de DataFrames usados para calcular os maiores índices.

    Returns:
        Tupla com número de usuários e número de itens.
    """
    full_data = pd.concat(dataframes, ignore_index=True)
    num_users = int(full_data["user_id_idx"].max()) + 1
    num_items = int(full_data["item_id_idx"].max()) + 1

    return num_users, num_items


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Executa uma época de treinamento.

    Args:
        model: Modelo PyTorch a ser treinado.
        dataloader: DataLoader da base de treino.
        optimizer: Otimizador usado para atualizar os pesos.
        loss_fn: Função de perda.
        device: Dispositivo onde os tensores serão processados.

    Returns:
        Perda média da época.
    """
    model.train()
    total_loss = 0.0

    for user_ids, item_ids, targets in dataloader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(user_ids, item_ids)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[list[int], list[float], float]:
    """Coleta targets, probabilidades previstas e perda média.

    Args:
        model: Modelo PyTorch avaliado.
        dataloader: DataLoader da base de validação ou teste.
        loss_fn: Função de perda.
        device: Dispositivo onde os tensores serão processados.

    Returns:
        Tupla contendo targets reais, probabilidades previstas e perda média.
    """
    model.eval()
    targets_list: list[int] = []
    probabilities: list[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for user_ids, item_ids, targets in dataloader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            targets = targets.to(device)

            logits = model(user_ids, item_ids)
            loss = loss_fn(logits, targets)
            probs = torch.sigmoid(logits)

            targets_list.extend(targets.cpu().numpy().astype(int).tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            total_loss += loss.item()

    return targets_list, probabilities, total_loss / len(dataloader)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float | None]:
    """Avalia o modelo e calcula métricas de classificação.

    Args:
        model: Modelo PyTorch avaliado.
        dataloader: DataLoader usado na avaliação.
        loss_fn: Função de perda.
        device: Dispositivo onde os tensores serão processados.

    Returns:
        Dicionário com métricas de classificação e loss.
    """
    y_true, y_proba, loss = collect_predictions(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
    )
    y_pred = [int(probability >= 0.5) for probability in y_proba]

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    metrics["loss"] = float(loss)

    return metrics


def save_outputs(
    model: nn.Module,
    metrics: dict[str, Any],
) -> None:
    """Salva o modelo treinado e o relatório de métricas.

    Args:
        model: Modelo treinado.
        metrics: Dicionário com métricas e parâmetros do treinamento.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODEL_FILE)

    REPORT_FILE.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def configure_mlflow() -> None:
    """Configura o MLflow Tracking e o Model Registry.

    A URI é lida da variável de ambiente MLFLOW_TRACKING_URI. Quando ela não
    está definida, o projeto usa um banco SQLite local como fallback.

    O mesmo backend é usado para tracking e registry para simplificar a execução
    local e a execução via Docker.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_params(
    args: argparse.Namespace,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> None:
    """Registra os parâmetros do treinamento no MLflow.

    Args:
        args: Argumentos de linha de comando usados no treinamento.
        num_users: Número total de usuários.
        num_items: Número total de itens.
        device: Dispositivo usado no treinamento.
    """
    mlflow.log_params(
        {
            "model_name": "recommender_net",
            "num_users": num_users,
            "num_items": num_items,
            "device": str(device),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout_rate": args.dropout_rate,
        }
    )


def log_metrics_to_mlflow(
    metrics: dict[str, float | None],
    prefix: str,
) -> None:
    """Registra métricas no MLflow com um prefixo.

    Args:
        metrics: Dicionário com métricas calculadas.
        prefix: Prefixo usado para diferenciar o tipo da métrica.
            Exemplo: "test_accuracy".
    """
    valid_metrics = {
        f"{prefix}_{metric_name}": metric_value
        for metric_name, metric_value in metrics.items()
        if metric_value is not None
    }

    mlflow.log_metrics(valid_metrics)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Treina o modelo usando early stopping.

    O melhor modelo, com base na menor loss de validação, é salvo em disco.
    O treinamento é interrompido quando a loss de validação não melhora
    durante a quantidade de épocas definida em patience.

    Args:
        model: Modelo PyTorch a ser treinado.
        train_loader: DataLoader da base de treino.
        validation_loader: DataLoader da base de validação.
        args: Argumentos de configuração do treinamento.
        device: Dispositivo usado no treinamento.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        validation_metrics = evaluate_model(
            model=model,
            dataloader=validation_loader,
            loss_fn=loss_fn,
            device=device,
        )
        validation_loss = validation_metrics["loss"]

        mlflow.log_metric("train_loss", train_loss, step=epoch)

        if validation_loss is not None:
            mlflow.log_metric("validation_loss", validation_loss, step=epoch)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break


def register_model_in_mlflow(
    model: nn.Module,
    metrics: dict[str, float | None],
) -> None:
    """Registra o modelo treinado no MLflow Model Registry.

    Esta função registra o modelo PyTorch no MLflow usando o backend configurado
    no projeto. O objetivo é criar uma entrada formal no Model Registry, com
    nome, versão, alias e tags de métricas.

    O modelo também já é salvo separadamente em `models/recommender_net.pt`,
    mas o registro no MLflow adiciona rastreabilidade entre experimento, versão
    do modelo e métricas finais.

    Args:
        model: Modelo PyTorch treinado.
        metrics: Métricas finais calculadas na base de teste.

    Raises:
        RuntimeError: Caso nenhuma versão registrada seja encontrada após o log.
    """
    model.eval()

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="recommender_net",
        registered_model_name=REGISTERED_MODEL_NAME,
        serialization_format="pickle",
        code_paths=["src"],
        metadata={
            "project": "tech-challenge-fase-02",
            "model_type": "pytorch_embedding_mlp",
            "target": "binary_positive_interaction",
        },
        await_registration_for=120,
    )

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")

    if not versions:
        raise RuntimeError("No registered model versions were found after logging.")

    latest_version = max(versions, key=lambda version: int(version.version))

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=latest_version.version,
    )

    for metric_name in ("roc_auc", "f1_score", "accuracy", "precision", "recall"):
        metric_value = metrics.get(metric_name)

        if metric_value is not None:
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=latest_version.version,
                key=metric_name,
                value=str(round(float(metric_value), 4)),
            )

    mlflow.log_param("registered_model_name", REGISTERED_MODEL_NAME)
    mlflow.log_param("registered_model_version", latest_version.version)
    mlflow.log_param("registered_model_alias", MODEL_ALIAS)

    print(
        "Model registered in MLflow Registry: "
        f"{REGISTERED_MODEL_NAME} v{latest_version.version} "
        f"alias={MODEL_ALIAS}"
    )


def run_training() -> None:
    """Executa o fluxo completo de treinamento e avaliação.

    O fluxo inclui:
        1. leitura dos argumentos;
        2. configuração do MLflow;
        3. carregamento dos dados;
        4. separação de treino e validação;
        5. criação dos DataLoaders;
        6. construção do modelo;
        7. treinamento com early stopping;
        8. avaliação na base de teste;
        9. salvamento de métricas e modelo;
        10. registro do modelo no MLflow Model Registry.
    """
    args = parse_args()
    set_seed(args.seed)
    configure_mlflow()

    train_data = load_dataset(TRAIN_FILE)
    test_data = load_dataset(TEST_FILE)
    train_data, validation_data = split_train_validation(train_data, args.seed)

    train_loader = create_dataloader(train_data, args.batch_size, shuffle=True)
    validation_loader = create_dataloader(
        validation_data,
        args.batch_size,
        shuffle=False,
    )
    test_loader = create_dataloader(test_data, args.batch_size, shuffle=False)

    num_users, num_items = get_model_sizes([train_data, validation_data, test_data])
    device = get_device(args.device)

    with mlflow.start_run(run_name=args.run_name):
        log_training_params(args, num_users, num_items, device)

        model = build_model(
            model_name="recommender_net",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
        ).to(device)

        train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            args=args,
            device=device,
        )

        loaded_state_dict = torch.load(
            MODEL_FILE,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(loaded_state_dict)

        loss_fn = nn.BCEWithLogitsLoss()
        test_metrics = evaluate_model(model, test_loader, loss_fn, device)

        final_report = {
            "model": "recommender_net",
            "metrics": test_metrics,
            "parameters": vars(args),
            "registry": {
                "registered_model_name": REGISTERED_MODEL_NAME,
                "alias": MODEL_ALIAS,
            },
        }

        save_outputs(model=model, metrics=final_report)
        log_metrics_to_mlflow(test_metrics, prefix="test")
        mlflow.log_artifact(str(REPORT_FILE))
        mlflow.log_artifact(str(MODEL_FILE))
        register_model_in_mlflow(model=model, metrics=test_metrics)

    print(f"PyTorch metrics saved to: {REPORT_FILE}")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    run_training()
