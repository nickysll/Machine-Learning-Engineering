"""Contrato de entrada: regras de qualidade aplicadas antes da inferência."""
import pandera.pandas as pa
from pandera import Check

# A fonte original tem categorias especiais em algumas variáveis; por isso,
# restringimos somente faixas com significado confirmado, sem inventar regras.
CREDIT_SCHEMA = pa.DataFrameSchema(
    {
        'ID': pa.Column(int, nullable=False, unique=True),
        'LIMIT_BAL': pa.Column(int, Check.gt(0), nullable=False),
        'AGE': pa.Column(int, Check.in_range(18, 100), nullable=False),
        'PAY_0': pa.Column(int, Check.in_range(-2, 9), nullable=False),
        'PAY_2': pa.Column(int, Check.in_range(-2, 9), nullable=False),
        'BILL_AMT1': pa.Column(int, nullable=False),
        'PAY_AMT1': pa.Column(int, Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=False,
    name='contrato_credit_scoring_v1',
)


def validate_credit_batch(df):
    """Lança exceção se ao menos uma regra do contrato falhar."""
    return CREDIT_SCHEMA.validate(df, lazy=True)
