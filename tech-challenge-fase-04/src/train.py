"""Treina baseline sem vazamento do holdout de produção."""
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from .config import DATA_DIR, MODEL_DIR, REPORT_DIR, FEATURES, TARGET, SEED
from .contracts import validate_credit_batch


def main():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    validate_credit_batch(train)
    validate_credit_batch(test)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED)),
    ])
    pipeline.fit(train[FEATURES], train[TARGET])
    predicted = pipeline.predict(test[FEATURES])
    probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        'n_train': len(train), 'n_test': len(test),
        'positive_rate_test': round(float(test[TARGET].mean()), 4),
        'accuracy': round(float(accuracy_score(test[TARGET], predicted)), 4),
        'precision': round(float(precision_score(test[TARGET], predicted, zero_division=0)), 4),
        'recall': round(float(recall_score(test[TARGET], predicted, zero_division=0)), 4),
        'f1': round(float(f1_score(test[TARGET], predicted, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(test[TARGET], probabilities)), 4),
        'average_precision': round(float(average_precision_score(test[TARGET], probabilities)), 4),
        'confusion_matrix': confusion_matrix(test[TARGET], predicted).tolist(),
        'features': FEATURES,
    }
    joblib.dump(pipeline, MODEL_DIR / 'baseline_credit.joblib')
    (REPORT_DIR / 'baseline_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print('Modelo e métricas salvos. Nenhuma decisão real de crédito foi executada.')

if __name__ == '__main__':
    main()
