import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from src.contracts import validate_credit_batch

@pytest.fixture
def valid_batch():
    return pd.DataFrame({
        'ID': [1, 2], 'LIMIT_BAL': [50000, 100000], 'AGE': [25, 40],
        'PAY_0': [0, 1], 'PAY_2': [0, -1], 'BILL_AMT1': [1000, 2500],
        'PAY_AMT1': [300, 600],
    })

def test_valid_batch_passes(valid_batch):
    assert len(validate_credit_batch(valid_batch)) == 2

@pytest.mark.parametrize('column,value', [
    ('AGE', 17), ('LIMIT_BAL', -10), ('PAY_AMT1', -1), ('PAY_0', 99),
])
def test_invalid_value_is_blocked(valid_batch, column, value):
    valid_batch.loc[0, column] = value
    with pytest.raises(SchemaErrors):
        validate_credit_batch(valid_batch)

def test_missing_value_is_blocked(valid_batch):
    valid_batch.loc[0, 'PAY_AMT1'] = None
    with pytest.raises(SchemaErrors):
        validate_credit_batch(valid_batch)

def test_duplicate_id_is_blocked(valid_batch):
    valid_batch.loc[1, 'ID'] = 1
    with pytest.raises(SchemaErrors):
        validate_credit_batch(valid_batch)
