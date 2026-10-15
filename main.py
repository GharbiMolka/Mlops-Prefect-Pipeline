"""
main.py
-------
Point d'entrée du pipeline de prédiction de churn client.
Permet d'exécuter les différentes étapes du pipeline via des arguments CLI.

Exemples d'utilisation :
    python main.py --step all
    python main.py --step prepare --data Churn_Modelling.csv
    python main.py --step train
    python main.py --step evaluate
    python main.py --step save
    python main.py --step load
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML - Prédiction de Churn Client"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["prepare", "train", "evaluate", "save", "load", "all"],
        help="Étape du pipeline à exécuter (défaut : all).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help="Chemin vers le fichier CSV des données.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Chemin du fichier modèle (.joblib) à sauvegarder/charger.",
    )
    args = parser.parse_args()

    if args.step == "prepare":
        x_train, x_test, y_train, y_test, scaler = prepare_data(args.data)
        print("Données préparées avec succès.")
        print(f"x_train shape: {x_train.shape} | x_test shape: {x_test.shape}")

    elif args.step == "train":
        x_train, x_test, y_train, y_test, scaler = prepare_data(args.data)
        model = train_model(x_train, y_train)
        print("Modèle entraîné avec succès.")

    elif args.step == "evaluate":
        x_train, x_test, y_train, y_test, scaler = prepare_data(args.data)
        model = train_model(x_train, y_train)
        evaluate_model(model, x_test, y_test)

    elif args.step == "save":
        x_train, x_test, y_train, y_test, scaler = prepare_data(args.data)
        model = train_model(x_train, y_train)
        save_model(model, args.model)

    elif args.step == "load":
        model = load_model(args.model)
        x_train, x_test, y_train, y_test, scaler = prepare_data(args.data)
        evaluate_model(model, x_test, y_test)

    elif args.step == "all":
        # Pipeline complet : préparation -> entraînement -> évaluation -> sauvegarde
        print("=== Étape 1 : Préparation des données ===")
        x_train, x_test, y_train, y_test, scaler = prepare_data(args.data)

        print("\n=== Étape 2 : Entraînement du modèle ===")
        model = train_model(x_train, y_train)

        print("\n=== Étape 3 : Évaluation du modèle ===")
        evaluate_model(model, x_test, y_test)

        print("\n=== Étape 4 : Sauvegarde du modèle ===")
        save_model(model, args.model)

        print("\nPipeline exécuté avec succès !")


if __name__ == "__main__":
    main()
