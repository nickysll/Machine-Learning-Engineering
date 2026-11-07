
from pathlib import Path

import pandas as pd

# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Dataset original de produção
df = pd.read_csv(DATA_DIR / "production_holdout.csv")


def simulate_drift(
    data: pd.DataFrame,
    limit_reduction: float,
    age_increase: int,
) -> pd.DataFrame:
    """Simula mudanças no perfil de crédito dos clientes."""

    simulated = data.copy()

    # Reduz o limite de crédito
    simulated["LIMIT_BAL"] = (
        simulated["LIMIT_BAL"] * (1 - limit_reduction)
    ).round().astype("int64")

    # Aumenta a idade
    simulated["AGE"] = (
        simulated["AGE"] + age_increase
    ).astype("int64")

    return simulated


# Cenários de simulação
scenarios = {
    "baseline": (0.00, 0),
    "moderate": (0.15, 3),
    "severe": (0.30, 7),
}

for name, (reduction, increase) in scenarios.items():

    simulated = simulate_drift(
        df,
        limit_reduction=reduction,
        age_increase=increase,
    )

    # Verificações básicas
    assert (simulated["LIMIT_BAL"] > 0).all()
    assert simulated["AGE"].between(18, 100).all()
    assert simulated["ID"].equals(df["ID"])

    output = DATA_DIR / f"production_{name}.csv"
    simulated.to_csv(output, index=False)

    print(f"\nCenário: {name}")
    print(f"Registros: {len(simulated)}")
    print(f"Limite médio: {simulated['LIMIT_BAL'].mean():,.2f}")
    print(f"Idade média: {simulated['AGE'].mean():.2f}")
    print(f"Arquivo: {output.name}")
