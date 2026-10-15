"""Atelier 3 : pipeline MLOps avec Prefect.

Lancement :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code
"""

import argparse
import subprocess
import sys
from pathlib import Path
import subprocess


from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)

# On ne cible QUE nos fichiers (sinon flake8/bandit scannent tout le venv)
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


def run(cmd):
    """Execute une commande et leve une erreur si elle echoue."""
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------ TASKS
@task(name="install_dependencies")
def install_dependencies():
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format_code")
def format_code():
    run([sys.executable, "-m", "black", *FILES])


@task(name="code_quality")
def code_quality():
    run([sys.executable, "-m", "flake8", "--max-line-length=100", *FILES])


@task(name="code_security")
def code_security():
    # -ll : ne remonte que les problemes de severite moyenne ou haute
    run([sys.executable, "-m", "bandit", "-q", "-ll", *FILES])


@task(name="unit_tests")
def unit_tests():
    tests = Path("tests")
    tests.mkdir(exist_ok=True)
    example = tests / "test_example.py"
    if not example.exists():
        example.write_text("def test_example():\n    assert 1 + 1 == 2\n")
    run([sys.executable, "-m", "pytest", "tests", "-v"])


@task(name="prepare_data")
def prepare_data_task():
    # ATTENTION : verifier l'ordre de retour de prepare_data (voir model_pipeline.py)
    x_train, x_test, y_train, y_test, _ = prepare_data(DATA_PATH)
    return x_train, x_test, y_train, y_test


@task(name="train_model")
def train_model_task(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="save_model")
def save_model_task(model):
    save_model(model, MODEL_PATH)


@task(name="load_model")
def load_model_task():
    return load_model(MODEL_PATH)


@task(name="evaluate_model")
def evaluate_model_task(model, x_test, y_test):
    result = evaluate_model(model, x_test, y_test)
    print(result)
    return result

@task
def run_api():
    subprocess.run(
        ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        check=True
    )



# ------------------------------------------------------------------ FLOWS
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    _, x_test, _, y_test = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, x_test, y_test)


@flow(name="all")
def all_flow():
    install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, x_test, y_test)


@flow(name="clean")
def clean_flow():
    for pattern in ("__pycache__", ".pytest_cache", MODEL_PATH):
        run(["rm", "-rf", pattern])

@flow(name="api")
def api_flow():
    run_api()

FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
    "install": install_flow,
    "clean": clean_flow,
    "api": api_flow,  # Ajouter cette ligne
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow",
        choices=FLOWS.keys(),
        default="all",
        help="Flow à exécuter : all, train, evaluate, code, install, clean, api",
    )
    args = parser.parse_args()

    FLOWS[args.flow]()