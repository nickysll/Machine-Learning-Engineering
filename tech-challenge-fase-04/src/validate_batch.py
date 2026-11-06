"""Demonstra que lotes inválidos são rejeitados, com evidência em JSON."""
import argparse
import json
import sys
from datetime import datetime, timezone
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from .config import DATA_DIR, REPORT_DIR
from .contracts import validate_credit_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', choices=['valid', 'invalid'], required=True)
    args = parser.parse_args()
    df = pd.read_csv(DATA_DIR / f'{args.batch}_batch.csv')
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'batch': args.batch, 'rows': len(df), 'status': 'approved', 'failures': []
    }
    try:
        validate_credit_batch(df)
    except (SchemaError, SchemaErrors) as exc:
        report['status'] = 'blocked'
        if isinstance(exc, SchemaErrors):
            report['failures'] = exc.failure_cases.fillna('NULL').astype(str).to_dict(orient='records')
        else:
            report['failures'] = [{'detail': str(exc)}]
    output = REPORT_DIR / f'validation_{args.batch}.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Status: {report['status'].upper()} | linhas={len(df)} | relatório={output}")
    if report['failures']:
        print(f"Total de ocorrências: {len(report['failures'])}")
    if report['status'] == 'blocked':
        sys.exit(1)  # intencional: impede processamento de lote inválido

if __name__ == '__main__':
    main()
