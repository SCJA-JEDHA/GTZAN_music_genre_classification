from sklearn import preprocessing
import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn

# Librosa (the mother of audio files)
import librosa
import librosa.display
from IPython import display
import warnings

from pathlib import Path


warnings.filterwarnings('ignore')

data = pd.read_csv(f'{general_path}/features_30_sec.csv')
data.head()


# Cyril data located 2 steps above :
general_path = 'C:\Users\Cyril\Documents\python\jedha\M11_projet\music_genre_classification\gtzan-dataset-music-genre-classification'
pca_sub_path = "\Data\PCA"
PCA_PATH = os.path.join(general_path, 'Data', 'PCA')


y = data['filename']
data2 = data.iloc[0:, 2:]
y = data2['label']

display(data2.head())
# X is w/o columns filename & label 
X = data2.loc[:, data2.columns != 'label']

#### NORMALIZE X ####
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)


#### PCA 10 COMPONENTS ####
from sklearn.decomposition import PCA
N_comp = 12
pca = PCA(n_components=N_comp)
principalComponents = pca.fit_transform(X)
PC_colnames = [f"princ_comp_{i+1}" for i in range(N_comp) ]
principalDf = pd.DataFrame(data = principalComponents, columns = PC_colnames)


# use with X_test:
PC_test = pca.transform(X_test)

# Sauvegarder l'instance PCA dans un fichier
import joblib
joblib.dump(pca, 'pca_model10.pkl')

# concatenate with target label
finalDf = pd.concat([principalDf, y, y2], axis = 1)

pca.explained_variance_ratio_
finalDf.head()
# 44.93 variance explained




from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle

# Définition du pipeline avec scaler et PCA
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=12))
])

# Entraînement du pipeline sur X
pipeline.fit(X)

# Transformation des données (optionnel)
X_pca = pipeline.transform(X)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Supposons que X est votre DataFrame initial (sans la colonne 'label')

# Définition du pipeline avec scaler et PCA
n_components = 12
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=n_components))
])

# Entraînement du pipeline sur X
pipeline.fit(X)

# Projection des données X dans le repère PCA
X_pca = pipeline.transform(X)

# Convertir en DataFrame pour plus de lisibilité (optionnel)
PC_colnames = [f"princ_comp_{i+1}" for i in range(n_components)]
principalDf = pd.DataFrame(data=X_pca, columns=PC_colnames)

print(principalDf.head())

# *-Récupérer la variance expliquée par chaque composante principale
explained_variance_ratio = pipeline.named_steps['pca'].explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

# Affichage graphique de la variance expliquée cumulée
plt.figure(figsize=(8,5))
plt.plot(range(1, n_components+1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.title('Variance expliquée cumulée en fonction du nombre de composantes principales')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Variance expliquée cumulée')
plt.xticks(range(1, n_components+1))
plt.grid(True)
plt.show()


# Sauvegarder le pipeline entraîné dans un fichier
with open('pipeline_pca.pkl', 'wb') as f:
    pickle.dump(pipeline, f)




### #######    
# Charger le pipeline depuis le fichier
with open('pipeline_pca.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Appliquer la transformation sur X_test
X_test_pca = loaded_pipeline.transform(X_test)


