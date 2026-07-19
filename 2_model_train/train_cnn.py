# Importation des bibliothèques #######
from calendar import EPOCH
import torch
from torchinfo import summary
import torchvision.transforms.v2 as transforms
from torchvision import datasets
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os, shutil
import pandas as pd
import plotly.express as px
import numpy as np
import torch.nn as nn
import torch.optim as optim
import librosa
import seaborn
import boto3
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import datetime
from torchviz import make_dot
import argparse
import subprocess
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import sys
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score

#######

####### VARIABLES ENVIRONNEMENT ET GENERALES #######
# Chargement des variables d'environnement
load_dotenv()
# Initialisation et création des répertoires s'ils n'existent pas #######
REP_KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification"
ZIP_FILE = "gtzan-dataset-music-genre-classification.zip"
# Locaux de base
PATH_BASE = "gtzan-dataset-music-genre-classification"
PATH_DATA = PATH_BASE + "/Data"
# Local des images RGB de spectrogrammes entiers
PATH_IMAGE = PATH_BASE + "/Data/images_original"
# Local des sons à partir desquels les spectrogrammes seront/ont été générés
PATH_SOUND = PATH_BASE + "/Data/genres_original"
# Local des images grey des spectrogrammes harmoniques
PATH_HARMO = PATH_BASE + "/Data/img_harmo"
# Local des images grey des spectrogrammes percussifs
PATH_PERCU = PATH_BASE + "/Data/img_percu"
# Le dataset nous facilitant les modules d'entrainements
PATH_DS = PATH_BASE + "/Data/features_30_sec.csv"
PATH_DS_SPLIT = PATH_BASE + "/Data/features_3_sec.csv"
#######

# Chargement des variables d'environnement
DATA_S3 = os.getenv("DATA_S3")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
#

# Variables d'entraînement ###
NUM_CLASSES = 10
####### FIN VARIABLES #######

####### FONCTIONS DE DIVERSES #######
#
# Fonction de suppression du dossier local ###
def delete_all_data():
    # Suppression du domaine de data LOCAL
    if os.path.exists(PATH_BASE):
        shutil.rmtree(PATH_BASE)
###
#
# Fonction de récupératio ndes données sur le S3 ###
def get_from_s3():
    try:
        # Instanciation client boto3
        print("Initialisation du client..")
        prefix = "music-database/"
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(DATA_S3) 
        print("..fait.")
        # Récuparation des datasetq
        print("[Récupération des données depuis {}]".format(DATA_S3))
        print("Téléchargement du dataset en cours..")
        bucket.download_file(prefix + PATH_DS, PATH_DS)
        bucket.download_file(prefix + PATH_DS_SPLIT, PATH_DS_SPLIT)
        print("..fait.")
        # Récupéation sons, images, harmo et percu
        print("Téléchargement des éléments en cours..")
        l_task = [str(k.key) for k in bucket.objects.filter(Prefix = prefix + PATH_SOUND)]
        l_task.extend([str(k.key) for k in bucket.objects.filter(Prefix = prefix + PATH_IMAGE)])
        l_task.extend([str(k.key) for k in bucket.objects.filter(Prefix = prefix + PATH_HARMO)])
        l_task.extend([str(k.key) for k in bucket.objects.filter(Prefix = prefix + PATH_PERCU)])
        # Affichage de la progression
        for s3_key in l_task:
            print(f"\r Progression : {((l_task.index(s3_key)/len(l_task))*100):.2f}% | {s3_key}.       ", end="", flush=True)
            path_out = os.path.dirname(s3_key).replace(prefix, "")
            file_out = s3_key.replace(prefix, "")
            os.makedirs(path_out, exist_ok = True)
            bucket.download_file(s3_key, file_out)
        print("\r Téléchargement des sons....[OK].                            \n", flush = True)
        print("[Données S3 récupérées]")
    # Gestion de l'exception
    except Exception as e:
        print(f"\nErreur lors du téléchargement : {e}.")
        delete_all_data()
###
#
# Fonction d'importation du dataset music/spectrogram/features (DEPRECATED, PASSAGE PAR S3)
def get_from_kaggle():
    # Vérification du dossier déjà existant
    if os.path.exists(PATH_BASE):
        print("Le dossier existe déjà, aucun téléchargement.")
    else:
        print(f"Téléchargement de {ZIP_FILE}..", end = "")
        subprocess.run(str.split(f"curl -L -o -q {ZIP_FILE} {REP_KAGGLE}"," ")) 
        print(f"..[OK].", "\n")
        # Extraction
        print(f"Extraction de l'archive {ZIP_FILE}..".format(), end = "")
        subprocess.run(str.split(f"unzip -o -q {ZIP_FILE} -d {PATH_BASE}"," ")) 
        print("..done.\n")
        # Suppression de l'archive
        print(f"Suppression {ZIP_FILE}..", end = "")
        subprocess.run(str.split(f"rm -rf {ZIP_FILE}"," "))
        print("..[OK].")
###
#
# Fonction de création des répertoires locaux de data ###
def rep_cnn_audio():
    # Si le répertoire de base n'existe pas
    if not os.path.exists(PATH_BASE):
        # Création du répertoire local
        print("Le dossier de base n'existe pas, création lancée.")
        print(f"Création de {PATH_BASE}..", end = "")
        os.makedirs(PATH_BASE)
        print("..[OK]")
        print(f"Création de {PATH_BASE}..", end = "")
        os.makedirs(PATH_DATA)
        print("..[OK]")
        # Récupération depuis le s3
        get_from_s3()
        # Spectrogrammes généraux
        print(f"Création de {PATH_IMAGE}..", end = "")
        os.makedirs(PATH_IMAGE)
        print("..[OK]")
        # Harmoniques
        print(f"Création de {PATH_HARMO}..", end = "")
        os.makedirs(PATH_HARMO)
        print("..[OK]")
        # Percussifs
        print(f"Création de {PATH_PERCU}..", end = "")
        os.makedirs(PATH_PERCU)
        print("..[OK]")
        # SOus-répertoires des spectrogrammes par genre
        l_subdirs = [d for d in os.listdir(PATH_SOUND)]
        print(f"Création des {len(l_subdirs)} en cours..", end="")
        for d in os.listdir(PATH_SOUND):
            os.makedirs(PATH_IMAGE + "/" + d)
            os.makedirs(PATH_HARMO + "/" + d)
            os.makedirs(PATH_PERCU + "/" + d)
        print("..[OK]")
        print("Dossier de base créé.")
        print("Lancement de la récupétation S3.")
        get_from_s3()
    else:
        print("Le dossier de base existe, poursuite de l'entrainement.")
###
#
# Fonction de génération de spectrogrammes généraux (RGB), harmonique (L) et percussifs (L)
def mel_hpss(in_path:str, out_path_b:str, out_path_h:str, out_path_p:str):
    # Chargement du son et application du trim
    y, sr = librosa.load(in_path)
    y, _ = librosa.effects.trim(y)
    # Définition des attributs standards sur
    n_fft = 2048        # Précision des détails du timbre
    hop_length = 512    # Résolution temporelle
    n_mels = 128        # Hauteur de la fréquence
    # Séparation harmonique/percussive
    y_full = y  # Conservation du général
    y_harmonic, y_percussive = librosa.effects.hpss(y) # Séparation harmonique/percussive
    # ndarray des spectrogrammes
    S_base = librosa.feature.melspectrogram(y=y_full, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels = n_mels) # Général
    S_h = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels = n_mels) # Harmonique
    S_p = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels = n_mels) # Percussifs
    # Conversion de la puissance du spectrogramme en dB
    S_DB = librosa.power_to_db(S_base, ref=np.max) # nous avons remplacé amplitude_to_db
    S_DB_h = librosa.power_to_db(S_h, ref=np.max)
    S_DB_p = librosa.power_to_db(S_p, ref=np.max)
    # Inversion des images (les images d'origines commencent par le 'haut')
    S_DB = np.flipud(S_DB)
    S_DB_h = np.flipud(S_DB_h)
    S_DB_p = np.flipud(S_DB_p)
    # Normalisation des images
    norm_S_DB = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())
    norm_S_DB_h = (S_DB_h - S_DB_h.min()) / (S_DB_h.max() - S_DB_h.min())
    norm_S_DB_p = (S_DB_p - S_DB_p.min()) / (S_DB_p.max() - S_DB_p.min())
    # Création des images en mémoire
    img_f = Image.fromarray((norm_S_DB * 255).astype(np.uint8))
    img_h = Image.fromarray((norm_S_DB_h * 255).astype(np.uint8), mode = "L")
    img_p = Image.fromarray((norm_S_DB_p * 255).astype(np.uint8), mode = "L")
    # Resize des images
    img_f = img_f.resize((512, 256), Image.Resampling.LANCZOS) # Plus grand pour le général -> [affichage]
    img_h = img_h.resize((256, 128), Image.Resampling.LANCZOS) # Plus modeste pour l'harmonique -> [CNN]
    img_p = img_p.resize((256, 128), Image.Resampling.LANCZOS) # Idem pour le percussif -> [CNN]
    # Sauvegarde dans les PATH locaux respectifs
    img_f.save(out_path_b)
    img_h.save(out_path_h)
    img_p.save(out_path_p)
###
#
# Génération itérative des trois spectrogrammes ###
def generate_spectrogrammes(ds:pd.DataFrame):
    try:
        print("Génération des images de spectrogrammes..", end = "")
        l_task = len(ds["filename"])

        for i, (fn, path_in, path_out_b, path_out_h, path_out_p) in enumerate(zip(ds["filename_wav"], ds["path_wav"], ds["path"], ds["path_harmo"], ds["path_percu"])):
            print(f"\rProgression : {(100*(i/l_task)):.2f}% | {fn}               ", end = "", flush = True)
            
            mel_hpss(path_in, path_out_b, path_out_h, path_out_p)

        print("\r..[OK]                                                                   ", flush = True)
    except Exception as e:
        print(f"\nErreur lors du téléchargement : {e}.")
###
#
# Fonction de suppression des répertoires d'images et recréation
def delete_imgs():
    # Suppression des répertoires d'images et recréation
    shutil.rmtree(PATH_IMAGE)
    shutil.rmtree(PATH_HARMO)
    shutil.rmtree(PATH_PERCU)
    # Spectrogrammes généraux
    os.makedirs(PATH_IMAGE)
    # Harmoniques
    os.makedirs(PATH_HARMO)
    # Percussifs
    os.makedirs(PATH_PERCU)
    # Sous-répertoires des spectrogrammes par genre
    for d in os.listdir(PATH_SOUND):
        os.makedirs(PATH_IMAGE + "/" + d)
        os.makedirs(PATH_HARMO + "/" + d)
        os.makedirs(PATH_PERCU + "/" + d)
###
#
####### FIN FONCTIONS #######
##########################################################################################################
####### CLASSES ########
#
# Classe CNN : Le réseau de neurones ###
class CNN_AudioSpectral(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_AudioSpectral, self).__init__()
        # BLOC CONV (FEATURES)
        self.features = nn.Sequential(
            # CONV1
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # CONV2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # CONV3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # FLATTEN
        self.flatten_size = 128 * 16 * 32
        # LE CLASSIFIER AVEC FCL
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # FCL1
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # FCL2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # CFCL FINALE
            nn.Linear(256, num_classes)
        )
    # Forward
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
###
# Classe CNN : Le réseau de neurones ###
class CNN_AudioSpectralV2(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_AudioSpectralV2, self).__init__()
        # BLOC CONV (FEATURES)
        self.features = nn.Sequential(
            # CONV1
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # CONV2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # CONV3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # # CONV4
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # FLATTEN
        #self.flatten_size = 256 * 8 * 16
        self.flatten_size = 128 * 16 * 32
        # LE CLASSIFIER AVEC FCL
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # FCL1
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # FCL2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # CFCL FINALE
            nn.Linear(256, num_classes)
        )
    # Forward
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
###
# Classe CNN : Le réseau de neurones ###
class CNN_AudioSpectralV3(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_AudioSpectralV3, self).__init__()
        # BLOC CONV (FEATURES)
        self.features = nn.Sequential(
            # CONV1
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # CONV2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # CONV3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # FLATTEN
        self.flatten_size = 256 * 16 * 32
        # LE CLASSIFIER AVEC FCL
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # FCL1
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # FCL2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # CFCL FINALE
            nn.Linear(256, num_classes)
        )
    # Forward
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
###
#
# Classe ModelCheckpoint : Vérification pour l'early stopping et la sauvegarde du modèle
class ModelCheckpoint:
    # filepath : chemin complet vers lequel le fichier sera sauvegardé
    # patience : délai d'époques à laquelle l'entrainement s'arrêtera
    # min_delta : delta de vérification de la condition surveillée (accuracy, loss, ou custom)
    # mode : min pour surveiller par exemple une meilleure loss (val loss qui baisse)
    # ou max pour surveiller par exemple une meilleure accuracy (val acc qui augmente)
    # ou une définition custom, par exemple surveiller une différence train_loss et val_loss qui reste en dessous de 0.2
    def __init__(self, filepath:str, patience:int=7, min_delta:float=0.001, mode:str="min"):
        self.filepath = filepath
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.count = 0
        self.best_score = None
        self.early_stop = False
    # model : à renseigner pour la sauvegarde
    # val_score : score à surveiller en min (meilleur min avec le delta) ou max
    def __call__(self, model, val_score):
        # Initialisation du val_score à l'appel de la classe
        if self.best_score is None:
            self.best_score = val_score
            torch.save(model.state_dict(), self.filepath)
            print("First save")
            return False
        # On reset le compteur de patience si on acquiet un meilleur val_score
        if (self.mode == "min" and val_score < (self.best_score - self.min_delta)) or \
            (self.mode == "max" and val_score > (self.best_score + self.min_delta)) or \
            (self.mode == "diff" and val_score <= 0.2) or \
            (self.mode == "none" and val_score == self.best_score):
            self.best_score = val_score
            self.count = 0
            torch.save(model.state_dict(), self.filepath)
            print("Save")
            return False
        # Si le score n'est pas meilleur aon augmente le count et s'il dépasse la patience on retoune True pour que l'entrainement déclenche l'early stopping
        else:
            print(f"{self.count}   ")
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
            return self.early_stop 
###
#
### --lr=1e-5
# Classe ImageDataset : Création de dataset spécialisé permettant de préparer une paire de tenseurs (harmonique, percussif) avec un label associé pour l'entrainement
class ImageDataset(Dataset):
    # Initialisation avec le dataset fourni et les transformations définies (les transformorations doivent être les mêmes pour les deux images)
    def __init__(self, ds:pd.DataFrame, mytransforms, mymapping:dict):
        self.ds = ds
        self.mytransforms = mytransforms
        self.mymapping = mymapping
    # Retourne la len du dataset
    def __len__(self):
        return len(self.ds)
    # getitem utilisé pour la création des dataloader
    def __getitem__(self, idx):
        # Récupération des images, les colonnes 3 et 4 du dataset retournent les chemins des images à charger
        image_h = Image.open(self.ds.iloc[idx, 3]) # Harmonique
        image_p = Image.open(self.ds.iloc[idx, 4]) # Percussif
        # Récupération des labels et encodage des labels
        label = self.ds.iloc[idx, 1]    # Recupération label
        label_enc = self.mymapping[label]   # Encode de label -> 0,1,2
        # Application des tranformations sur les images -> transformations diverses -> tensor
        image_h = self.mytransforms(image_h)
        image_p = self.mytransforms(image_p)
        # Retourne la paire de tenseurs avec le label associé
        return torch.cat((image_h, image_p), dim=0), label_enc
###
####### FIN CLASSES #######
###########################################################################################################
####### FONCTIONS LIEES AU PROCESS D'ENTRAINEMENT #######
#
# Fonction de préparation de dataset ###
def prepare_dataset() -> pd.DataFrame:
    # Exécution de la création si non existant : création des répertoires, sous-répertoires, récupération des fichier audio et génération des spectrogrammes
    rep_cnn_audio()
    # Préparation du dataset, en ignorant jazz.00054.wav (corrompu), puis tri par labels
    ds = pd.read_csv(PATH_DS, encoding = "utf-8")
    ds = ds[ds["filename"] != "jazz.00054.wav"]
    ds = ds.sort_values(by = "label", ascending = True)
    # Suppression des colonnes non-utilisées
    c_to_drop = [c for c in ds.columns if c not in ["filename", "label"]]
    ds = ds.drop(columns = c_to_drop)
    # Modification/Création des colonnes utilisées, chemins des fichiers images
    ds["filename_wav"] = ds["filename"]
    ds["filename"] = [str.replace(c, ".wav", ".png").replace(".0", "0") for c in ds["filename"]]
    ds["path_harmo"] = [PATH_HARMO + "/" + c + "/" + f for c, f in zip(ds["label"], ds["filename"])]
    ds["path_percu"] = [PATH_PERCU + "/" + c + "/" + f for c, f in zip(ds["label"], ds["filename"])]
    ds["path"] = [PATH_IMAGE + "/" + c + "/" + f for c, f in zip(ds["label"], ds["filename"])]
    ds["path_wav"] = [PATH_SOUND + "/" + c + "/" + f for c, f in zip(ds["label"], ds["filename_wav"])]
    # Retour du ds
    return ds
###
#
# Fonction retournant une transformation CUSTOM pour le train et val ###
def trainval_mytransform():
  mytransform = transforms.Compose([
    # Transformations pour la data augmentation sous l'entrainement
    transforms.RandomHorizontalFlip(),
    # totensor() sous v2
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean = [0.5], std = [0.5])
  ])
  return mytransform
###
#
# Fonction de transformation en tensor normalisé ###
def test_mytransform():
  mytransform = transforms.Compose([
    # Aucune transormation en dehors de totensor sous v2 et normalisation
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean = [0.5], std = [0.5])
  ])
  return mytransform
###
#
# Fonction de train
def train_process(
    model:nn.Module, 
    train_loader:DataLoader, 
    val_loader:DataLoader, 
    criterion:nn.CrossEntropyLoss, 
    optimizer:optim.AdamW|optim.Adam, 
    epochs:int=200, 
    patience:int=7) -> dict:

    d_early_stop = {

        "min": ModelCheckpoint(
            filepath = "best_model_gtzan.pth",
            patience = patience,
            min_delta = 0.001,
            mode = "min"
        ),

        "max" : ModelCheckpoint(
            filepath = "best_model_gtzan.pth",
            patience = patience,
            min_delta = 0.01,
            mode = "max"
        ),

        "diff" : ModelCheckpoint(
            filepath = "best_model_gtzan.pth",
            patience = patience,
            mode = "diff"
        ),

        "none" : ModelCheckpoint(
            filepath = "best_model_gtzan.pth",
            patience = patience,
            mode = "none"
        )
    }

    mycallback = d_early_stop[EARLY_STOP]

    # Dictionaire permettant de mémoriser la train_loss, val_loss, train_acc, val_acc
    d_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'diff_loss' : []}
    # Sécurité pour ne pas descendre à un loss nulle
    min_loss_threshold = 1e-5 

    # On boucle sur les epochs pour l'entrainement
    for epoch in range(epochs):
        
        model.train()  # On passe le modèle en mode d'entrainement (modification des gradients)
        model = model.to(DEVICE) # On passe le modèle sur le GPU
        total_loss, correct = 0, 0  # Initialisation de la loss totale et des prédictions correctes

        # Training loop
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()  # Reset des gradients avant chaque batch
            outputs = model(inputs).squeeze()  # Forward pass
            loss = criterion(outputs, labels)  # Calcul de la train_loss
            loss.backward()  # Rétropropagation (calcul des gradients)
            optimizer.step()  # Mise à jour des paramètres du modèle grace à l'optimizer

            total_loss += loss.item()  # Accumulation de la loss
            correct += (torch.argmax(outputs,dim=1) == labels).sum().item()  # On compte les prédictions correctes

        # Calcul de la loss moyenne pour TRAIN -> train_loss pour l'époque, idem pour l'accuracy
        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        # CONDITION DE SÉCURITÉ : Arrêt si la train_loss approche de zéro
        if train_loss < min_loss_threshold:
            torch.save(model.state_dict(), FILEPATH)
            print(f"\n--- Arrêt de sécurité : Train Loss quasi nulle ({train_loss:.6f}) à l'époque {epoch+1} ---")
            break

        # Phase de validation (on vérifie la qualité de l'entrainement sur des données non-vues, on ne calcule pas les gradients)
        model.eval()  # Passage en mode évaluation (sans calcul de gradient)
        val_loss, val_correct = 0, 0
        with torch.no_grad():  # Process d'inférence sans calcul de gradients
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs).squeeze()  # Inférence
                loss = criterion(outputs, labels)  # Calcul de la val_loss
                val_loss += loss.item()  # Accumulation del a val_loss
                val_correct += (torch.argmax(outputs,dim=1) == labels).sum().item()  # On compte les prédictions correctes

            # Calcul de la loss moyenne pour VALIDATION -> val_loss pour l'époque, idem pour l'accuracy
            val_loss /= len(val_loader)
            val_acc = val_correct / len(val_loader.dataset)

        diff_loss = val_loss - train_loss

        # Enregistrement des métriques d'entrainement dans MLFLOW
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("loss_diff", diff_loss, step=epoch)

        # Enregistrement des métriques d'entrainement dans le dictionnaire
        d_history['loss'].append(train_loss)
        d_history['val_loss'].append(val_loss)
        d_history['accuracy'].append(train_acc)
        d_history['val_accuracy'].append(val_acc)
        d_history['diff_loss'].append(diff_loss)
                
        # Step du scheduler : si définit, le scheduler va executer une opération supplémentaire sur le LR en fonction de la surveillance d'entrainement
        #scheduler.step(val_loss)
        
        if EARLY_STOP == "min":
            score_to_send = val_loss
        elif EARLY_STOP == "max":
            score_to_send = val_acc
        elif EARLY_STOP == "diff":
            score_to_send = diff_loss
        else:
            score_to_send = 1
        
        # Surveillance early stop sur la val_loss : si elle ne descend plus passé un délais de patience, on arrête l'entrainement
        if mycallback(model, score_to_send):
            print(f"Early stopping à l'époque {epoch}")
            break

        # Affichage de l'amélioration
        print(f"\nEpoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} Diff Loss {diff_loss:.4f}         ", end = "")

    return d_history
###
#
# Fonction de graphique loss val et diff_loss
def mlflowg_historic(d_history:dict):
    
    # Graphique LOSS
    fig_loss, ax1 = plt.subplots(figsize=(10, 4))
    # Ajout des courbes
    ax1.plot(d_history["loss"], label="Training loss", color="blue", linestyle="-")
    ax1.plot(d_history["val_loss"], label="Validation loss", color="green", linestyle="-")
    # Titre et axes
    ax1.set_title('Training Loss et Val Loss à travers les époques')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross Entropy')
    ax1.legend()
    # Log MLFLOW
    mlflow.log_figure(fig_loss, "history_loss.png")
    # Fermeture
    plt.close(fig_loss)

    # Graphique ACCURACY
    fig_acc, ax2 = plt.subplots(figsize=(10, 4))
    # 2. Ajout des courbes
    ax2.plot(d_history["accuracy"], label="Training Acc", color="red", linestyle="-")
    ax2.plot(d_history["val_accuracy"], label="Validation Acc", color="purple", linestyle="-")
    # Titre et axes
    ax2.set_title('Training Accuracy et Val Accuracy à travers les époques')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    # Log MLFLOW
    mlflow.log_figure(fig_acc, "history_acc.png")
    # Fermeture
    plt.close(fig_loss)

    # Graphique DIFF LOSS
    fig_acc, ax3 = plt.subplots(figsize=(10, 4))
    # Ajout de la courbe
    ax3.plot(d_history["diff_loss"], label="Training Acc", color="red", linestyle="-")
    # Titre et axes
    ax3.set_title('Différence entre train_loss et val_loss à travers les époques')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Différence de loss')
    ax3.legend()
    # Log MLFLOW
    mlflow.log_figure(fig_acc, "diff_loss.png")
    # Fermeture
    plt.close(fig_acc)
###
####### FIN FONCTIONS D'ENTRAINEMENT #######
##########################################################################################################

####### MAIN PROCESS #######

# example of launch : 
#python train_cnn.py --model_version=v2 --early_stop=none --lr=1e-4
#
if __name__ == "__main__":
    # Parser des arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=str, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--experiment_name", type=str, default="audio_classifier")
    parser.add_argument("--model_name", type=str, default="audio_classifier")
    parser.add_argument("--model_version", type=str, default="v1")
    parser.add_argument("--optimizer_name", type=str, default="AdamW")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--filepath", type=str, default="best_model_gtzan.pth")
    parser.add_argument("--mlflow_model_name", type=str, default="baseline_cnn_audio_classifier")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--early_stop", type=str, default="min") # case diff : 
    args = parser.parse_args()

    # Récupération des arguments
    BATCH_SIZE = args.batch_size
    TEST_SIZE = args.test_size
    VAL_SIZE = args.val_size
    RANDOM_STATE = args.random_state
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    NUM_CLASSES = 10
    EXPERIMENT_NAME = args.experiment_name
    MODEL_NAME = args.model_name
    MODEL_VERSION = args.model_version
    OPTIMIZER_NAME = args.optimizer_name
    N_EPOCHS = args.n_epochs
    FILEPATH = args.filepath
    PATIENCE = args.patience
    MLFLOW_MODEL_NAME = args.mlflow_model_name
    EARLY_STOP=args.early_stop

    # Dictionnaire des modèle pour adresser un paramètre de modèle spécifique
    # Si on souhaite créer une autre classe de nn, on ajoute la classe dans l'espace dessus
    # puis on l'ajoute au dictionnaire pour qu'il soit pris en compte dans les paramètres
    d_model_version = {
        "v1" : CNN_AudioSpectral(num_classes=NUM_CLASSES),
        "v2" : CNN_AudioSpectralV2(num_classes=NUM_CLASSES),
        "v3" : CNN_AudioSpectralV3(num_classes=NUM_CLASSES)
    }

    # Vérification de l'existence du répertoire et récupération si non présent
    rep_cnn_audio()

    # Création du dataset
    ds = prepare_dataset()
    # Création d'un mapping pour encoder les classes
    unique_labels = list(ds["label"].unique().tolist())
    mapping_LI = {l : unique_labels.index(l) for l in unique_labels}
    # Création du reverse mapping pour les rapports de prédictions sur le test
    reverse_LI = {v : k for v, k in enumerate(mapping_LI)}

    print("Mapping definition as : {}".format(mapping_LI))

    # Splitting with stratify
    ds_train, ds_test = train_test_split(ds, test_size = TEST_SIZE, random_state = RANDOM_STATE, stratify = ds["label"])
    ds_train, ds_val = train_test_split(ds_train, test_size = VAL_SIZE, random_state = RANDOM_STATE, stratify = ds_train["label"])

    # Affichage des tailles des ds
    print("Taille du train : ", ds_train.shape)
    print("Taille du val : ", ds_val.shape)
    print("Taille du test : ", ds_test.shape)

    # Instancing the ImageDataset for train and val
    ids_train = ImageDataset(ds_train, mytransforms = trainval_mytransform(), mymapping = mapping_LI)
    ids_val = ImageDataset(ds_val, mytransforms = trainval_mytransform(), mymapping = mapping_LI)
    ids_test = ImageDataset(ds_test, mytransforms = test_mytransform(), mymapping = mapping_LI)

    # Création des DataLoader que nous utiliserons pour l'entrainement et le test
    train_loader = DataLoader(ids_train, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
    val_loader = DataLoader(ids_val, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(ids_test, batch_size = BATCH_SIZE)

    # Génération des spectrogrammes
    if not os.path.exists(PATH_BASE):
        generate_spectrogrammes(ds)
    
    
    # Initialisation CPU/GPU et vérification pour continuer
    DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    if DEVICE == "cpu":
        print("Le device cuda n'est pas disponible pour l'entraînement, veuillez passer en GPU.")
        print("Fin de l'entraintement.")
        sys.exit()

    # Tracking avec MLFLOW
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog(log_models=False)

    # Instanciation du modèle, du criterion et de l'optimizer
    model_cnn = d_model_version[MODEL_VERSION]
    criterion = nn.CrossEntropyLoss()

    d_optimizer = {
        "Adam" : optim.Adam(model_cnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        "AdamW" : optim.AdamW(model_cnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    }
    optimizer = d_optimizer[OPTIMIZER_NAME]

    full_dt = datetime.datetime.now()
    soft_dt = full_dt.strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name="CNN_audio_classifier_" + soft_dt):

        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": N_EPOCHS,
            "patience": PATIENCE,
            "num_classes": NUM_CLASSES,
            "device": str(DEVICE)
        })

        # Entrainement : PROCESS QUI S'AFFICHERA PRINCIPALEMENT
        d_history = train_process(
            model=model_cnn,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=N_EPOCHS,
            patience=PATIENCE)

        # PREDICTIONS ET LOGS MLFLOW
        # Instanciation du meilleur modèle sauvegaré
        best_model = d_model_version[MODEL_VERSION]
        best_model.load_state_dict(torch.load(FILEPATH))
            
        # Envoi en GPU pour prédiction du test_loader
        best_model.to(DEVICE)
        best_model.eval()
            
        Y_true, Y_pred = [], []
            
        with torch.no_grad():

            for batch_X, batch_Y in test_loader:

                batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
                output = best_model(batch_X)

                _, predicted = torch.max(output, 1)
                Y_true.extend(batch_Y.cpu().numpy())
                Y_pred.extend(predicted.cpu().numpy())

        Y_true = np.array(Y_true)
        Y_pred = np.array(Y_pred)
        #####

        # Création des scores #####
        test_acc = accuracy_score(Y_true, Y_pred)
        test_f1 = f1_score(Y_true, Y_pred, average="macro")
        # MLFLOW LOG des scores
        # LOG MLFLOW Divers #####
        mlflow.log_metric("TEST_accuracy", test_acc)
        mlflow.log_metric("TEST_model_f1", test_f1)
        #####
            
        # MLFLOW LOG du meilleur modèle
        input_dummy = torch.randn(1, 2, 128, 256).to(DEVICE)
        with torch.no_grad():
            output_dummy = best_model(input_dummy)
        # Signature pour enregisterer le modèle
        signature = infer_signature(input_dummy, output_dummy)
        # MLGLOW Modèle
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            name=MLFLOW_MODEL_NAME,
            registered_model_name=MLFLOW_MODEL_NAME,
            signature=signature, # Enregistre le format d'entrée/sortie
            input_example=input_dummy[:1].cpu().numpy() # Enregistre une image spectrale pour l'interface visuelle [History]
        )
        #####

        # Création de CMD #####
        class_names = list(mapping_LI.keys())
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay.from_predictions(
            Y_true, 
            Y_pred, 
            display_labels=class_names,
            normalize="true",
            colorbar = "false",
            cmap='Blues', 
            xticks_rotation=45,
            values_format=".0%",
            ax=ax
        )
        plt.title("Matrice de confusion de test du CNN")
        # LOG MLFLOW CMD
        mlflow.log_figure(fig, "confusion_matrix_audio_classifier_cnn.png")
        #####

        # Création du graphe #####
        y = model_cnn(torch.randn(1, 2, 128, 256).to(DEVICE))
        dot = make_dot(y, params = dict(model_cnn.named_parameters()))
        img = dot.render("logged_model_graph", format="png")
        # MLFLOW LOG du graphe
        mlflow.log_artifact(img)
        #####

        # Création du summary #####
        model_summary = summary(model_cnn, input_size = (1, 2, 128, 256))
        with open("model_summary.txt", "w") as file:
            file.write(str(model_summary))
        # MLFLOW LOG du summary
        mlflow.log_artifact("model_summary.txt")
        #####

        # MLFLOW LOG LOSS ACC
        mlflowg_historic(d_history)
        #####