"""
model_pipeline.py
------------------
Fonctions modularisées du pipeline de Machine Learning pour la prédiction
du churn client (Churn_Modelling.csv).

Étapes couvertes :
    - prepare_data()   : chargement + prétraitement des données
    - train_model()    : entraînement du modèle
    - evaluate_model()  : évaluation des performances
    - save_model()      : sauvegarde du modèle entraîné (joblib)
    - load_model()      : chargement d'un modèle sauvegardé
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# 1. Préparation des données
# ---------------------------------------------------------------------------
def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 1):
    """
    Charge le fichier CSV et prétraite les données pour l'entraînement.

    Étapes :
        - Chargement du CSV
        - Encodage de la colonne 'Gender' (LabelEncoder)
        - Suppression des colonnes non pertinentes
          ('Surname', 'Geography', 'RowNumber', 'CustomerId')
        - Séparation features (X) / cible (y = 'Exited')
        - Split train / test
        - Standardisation des features (StandardScaler)

    Args:
        filepath (str): chemin vers le fichier CSV (Churn_Modelling.csv).
        test_size (float): proportion du jeu de test (défaut : 0.2).
        random_state (int): graine aléatoire pour la reproductibilité.

    Returns:
        tuple: (x_train_scaled, x_test_scaled, y_train, y_test, scaler)
    """
    df = pd.read_csv(filepath)

    # Encodage de la variable catégorielle 'Gender'
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes non utiles pour le modèle
    columns_to_drop = ["Surname", "Geography"]
    df = df.drop(columns_to_drop, axis=1)

    # Séparation features / cible
    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    x = x.drop(columns=["RowNumber", "CustomerId"])

    # Split train / test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Standardisation
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)  # transform (pas fit) sur le test

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# 2. Entraînement du modèle
# ---------------------------------------------------------------------------
def train_model(x_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Entraîne un RandomForestClassifier sur les données d'entraînement.

    Args:
        x_train: features d'entraînement.
        y_train: cible d'entraînement.
        n_estimators (int): nombre d'arbres de la forêt (défaut : 100).
        random_state (int): graine aléatoire.

    Returns:
        RandomForestClassifier: le modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3. Évaluation du modèle
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test):
    """
    Évalue les performances du modèle sur le jeu de test.

    Args:
        model: modèle entraîné.
        x_test: features de test.
        y_test: cible de test.

    Returns:
        dict: contient l'accuracy, le rapport de classification et
              la matrice de confusion.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("Classification report:")
    print(report)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------------------------
# 4. Sauvegarde du modèle
# ---------------------------------------------------------------------------
def save_model(model, filepath: str = "classifier.joblib"):
    """
    Sauvegarde le modèle entraîné sur disque avec joblib.

    Args:
        model: modèle entraîné à sauvegarder.
        filepath (str): chemin de destination (défaut : 'classifier.joblib').
    """
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans : {filepath}")


# ---------------------------------------------------------------------------
# 5. Chargement du modèle
# ---------------------------------------------------------------------------
def load_model(filepath: str = "classifier.joblib"):
    """
    Charge un modèle préalablement sauvegardé.

    Args:
        filepath (str): chemin du fichier modèle à charger.

    Returns:
        Le modèle chargé.
    """
    model = joblib.load(filepath)
    print(f"Modèle chargé depuis : {filepath}")
    return model
