
FROM python:3.12-slim

WORKDIR /app

# Éviter les fichiers .pyc et activer les logs immédiats
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier les fichiers nécessaires à l'application
COPY app.py .
COPY main.py .
COPY model_pipeline.py .
COPY classifier.joblib .
COPY Churn_Modelling.csv .

# Exposer le port FastAPI
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
