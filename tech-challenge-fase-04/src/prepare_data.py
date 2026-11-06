"""Baixa dataset público da UCI, valida e separa treino/teste/produção."""
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from .config import DATA_DIR, TARGET, SEED
from .contracts import validate_credit_batch


def main():
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    # Compatibilidade com possíveis nomes originais X1..X23.
    mapping = {
        'X1':'LIMIT_BAL', 'X5':'AGE', 'X6':'PAY_0', 'X7':'PAY_2',
        'X12':'BILL_AMT1', 'X18':'PAY_AMT1'
    }
    X = X.rename(columns=mapping)
    X.columns = [str(col).strip() for col in X.columns]
    if len(X) < 5000:
        raise ValueError('Dataset deve ter pelo menos 5.000 registros.')
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if y.shape[1] != 1:
        raise ValueError(f'Esperado um único target; recebido {list(y.columns)}')
    frame = X.copy()
    frame.insert(0, 'ID', range(1, len(frame) + 1))
    frame[TARGET] = pd.to_numeric(y.iloc[:, 0], errors='raise').astype(int)
    if not frame[TARGET].isin([0,1]).all():
        raise ValueError('Target não é binário.')
    validate_credit_batch(frame)
    # 60% treino, 20% teste, 20% holdout de produção (não visto no treino).
    reference, production = train_test_split(
        frame, test_size=0.20, stratify=frame[TARGET], random_state=SEED)
    train, test = train_test_split(
        reference, test_size=0.25, stratify=reference[TARGET], random_state=SEED)
    for name, part in [('train', train), ('test', test), ('production_holdout', production)]:
        part.to_csv(DATA_DIR / f'{name}.csv', index=False)
        print(f'{name}: {len(part)} linhas; inadimplência={part[TARGET].mean():.2%}')
    # Lote válido e lote com erros propositalmente inseridos.
    valid_batch = test.head(20).copy().reset_index(drop=True)
    valid_batch.to_csv(DATA_DIR / 'valid_batch.csv', index=False)
    invalid = valid_batch.copy()
    invalid.loc[0, 'AGE'] = 15
    invalid.loc[1, 'LIMIT_BAL'] = -500
    invalid.loc[2, 'PAY_AMT1'] = None
    invalid.loc[3, 'ID'] = invalid.loc[4, 'ID']
    invalid.to_csv(DATA_DIR / 'invalid_batch.csv', index=False)
    print('Dados preparados. Fonte: UCI ML Repository, dataset 350, CC BY 4.0.')

if __name__ == '__main__':
    main()
