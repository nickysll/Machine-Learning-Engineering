from datetime import UTC, datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT_DIR = "/opt/airflow/project"


with DAG(
    dag_id="medical_text_training_pipeline",
    description="Medical text classification training pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    tags=["machine-learning", "phase-3"],
) as dag:

    download_data = BashOperator(
        task_id="download_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python scripts/download_data.py"
        ),
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python scripts/validate_data.py"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/training/train.py"
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/training/evaluate.py"
        ),
    )

    save_model = BashOperator(
        task_id="save_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "cp models/model.joblib "
            "models/model_production.joblib"
        ),
    )

    (
        download_data
        >> validate_data
        >> train_model
        >> evaluate_model
        >> save_model
    )
    