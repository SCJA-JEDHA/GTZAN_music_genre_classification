import io
import pandas as pd
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning
)  # to avoid deprecation warnings
# pip install plotly

# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.io as pio

def charger_csv_google_drive(file_id):
    """Télécharge un fichier CSV depuis Google Drive en contournant

    l'avertissement de virus de Google si nécessaire, puis le charge dans
    Pandas.
    """
    url = "https://docs.google.com/uc?export=download"

    # Création d'une session pour conserver les cookies (nécessaire pour le token de confirmation)
    session = requests.Session()
    reponse = session.get(url, params={"id": file_id}, stream=True)

    # Vérification si Google Drive demande une confirmation pour le téléchargement
    token = None
    for clé, valeur in reponse.cookies.items():
        if clé.startswith("download_warning"):
            token = valeur
            break

    # Si un token de confirmation est requis, on refait la requête avec ce token
    if token:
        params = {"id": file_id, "confirm": token}
        reponse = session.get(url, params=params, stream=True)

    # Si le statut est 200 (OK), on charge le contenu directement dans Pandas
    if reponse.status_code == 200:
        print("Connexion réussie ! Chargement des données dans Pandas...")

        # io.BytesIO permet à Pandas de lire les données directement depuis la mémoire
        # sans avoir à créer de fichier physique sur votre disque dur.
        donnees_binaires = io.BytesIO(reponse.content)

        # Chargement dans le DataFrame Pandas
        # (Vous pouvez ajouter l'argument sep=',' ou sep=';' selon votre CSV)
        df = pd.read_csv(donnees_binaires)
        return df
    else:
        print(
            f"Erreur lors du téléchargement. Code statut : {reponse.status_code}"
        )
        return None


# ID extrait de votre lien Google Drive
ID_FICHIER = "1chgEVQaY1hrUYZdlyI2cvkdF6QV-KqVg"

# Appel de la fonction
df = charger_csv_google_drive(ID_FICHIER)

# Separate target variable Y from features X
target_variable = "label"
X = df.drop([target_variable,"filename", "length"], axis=1)
Y = df.loc[:, target_variable]

# Divide dataset Train set & Test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=0, stratify=Y
)

# 1. Initialiser le StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Label encoding train
encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)

# Preprocessings on test set
X_test = scaler.transform(X_test)

# Label encoding test
Y_test = encoder.transform(Y_test)

#Model RandomForest
model = RandomForestClassifier(random_state=42, max_depth=8, min_samples_leaf= 15, min_samples_split= 20, n_estimators= 150)
model.fit(X_train, Y_train)

# Predictions on training set
Y_train_pred = model.predict(X_train)

# Predictions on test set
Y_test_pred = model.predict(X_test)

# Print scores
print("accuracy on training set : ", accuracy_score(Y_train, Y_train_pred))
print("accuracy on test set : ", accuracy_score(Y_test, Y_test_pred))