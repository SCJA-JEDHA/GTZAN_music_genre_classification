# API model predict 

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from enum import Enum
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Literal, List, Union, Any, Dict
import torch
import torchvision.transforms.v2 as transforms
from torchinfo import summary
from PIL import Image
import mlflow.pytorch
import mlflow
from dotenv import load_dotenv
import numpy as np
import os
import io
from contextlib import redirect_stdout
import base64


# Load environment variables from the .env file (if present)
load_dotenv()

# import model : 
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
REGISTERED_MODEL_NAME = "BEST_cnn_audio_classifier"
DICT_LABEL = ""
#DS_PRED = pd.DataFrame("./ds_predictions.csv", encoding = "utf-8")

### 
# Here you can define some configurations 
###
# class for model_dict:
class ItemModel(BaseModel):
    data: Dict[str, Any]


# class features 
class NumFeatures(BaseModel):
    # list of numeric features:
    filename : str
    length : int
    chroma_stft_mean : Union[int, float]
    chroma_stft_var : Union[int, float]
    rms_mean    : Union[int, float]
    rms_var : Union[int, float] 
    spectral_centroid_mean: Union[int, float]
    spectral_centroid_var :Union[int, float]
    spectral_bandwidth_mean : Union[int, float]
    spectral_bandwidth_var : Union[int, float] 
    rolloff_mean : Union[int, float]
    rolloff_var : Union[int, float] 
    zero_crossing_rate_mean : Union[int, float]
    zero_crossing_rate_var : Union[int, float]
    harmony_mean : Union[int, float]
    harmony_var : Union[int, float]
    perceptr_mean : Union[int, float]
    perceptr_var : Union[int, float]
    tempo : Union[int, float]
    mfcc1_mean : Union[int, float]
    mfcc1_var : Union[int, float]
    mfcc2_mean : Union[int, float]
    mfcc2_var : Union[int, float]
    mfcc3_mean : Union[int, float]
    mfcc3_var : Union[int, float]
    mfcc4_mean : Union[int, float]
    mfcc4_var : Union[int, float]
    mfcc5_mean : Union[int, float]
    mfcc5_var : Union[int, float]
    mfcc6_mean : Union[int, float]
    mfcc6_var : Union[int, float]
    mfcc7_mean : Union[int, float]
    mfcc7_var : Union[int, float]
    mfcc8_mean : Union[int, float]
    mfcc8_var : Union[int, float]
    mfcc9_mean : Union[int, float]
    mfcc9_var : Union[int, float]
    mfcc10_mean : Union[int, float]
    mfcc10_var : Union[int, float]
    mfcc11_mean : Union[int, float]
    mfcc11_var : Union[int, float]
    mfcc12_mean : Union[int, float]
    mfcc12_var : Union[int, float]
    mfcc13_mean : Union[int, float]
    mfcc13_var : Union[int, float]
    mfcc14_mean : Union[int, float]
    mfcc14_var : Union[int, float]
    mfcc15_mean : Union[int, float]
    mfcc15_var : Union[int, float]
    mfcc16_mean : Union[int, float]
    mfcc16_var : Union[int, float]
    mfcc17_mean : Union[int, float]
    mfcc17_var : Union[int, float]
    mfcc18_mean : Union[int, float]
    mfcc18_var : Union[int, float]
    mfcc19_mean : Union[int, float]
    mfcc19_var : Union[int, float]
    mfcc20_mean : Union[int, float]
    mfcc20_var : Union[int, float]
    label : str
    
# class image_coord 
    
class ImageCoord(BaseModel):
    # list of numeric features:
    
    #la liste est du type : 
    # [[],[],[]] : chaque liste a une taille de IMAGE_N elements float
    
    coord: List[Union[int, float]]

    @field_validator('coord')
    def check_length(cls, v):
        # if len(v) != IMAGE_N:
        #     raise ValueError(f'coord doit avoir une longueur de {IMAGE_N}, mais a {len(v)}')
        return v

    def __repr__(self):
        return f"ImageCoord(coord_len={len(self.coord)})"

class PredictionRequest_f(BaseModel):
    """ contient model_name & NumFeatures"""
    model_name: str
    num_features: NumFeatures

class PredictionRequest_i(BaseModel):
    """ contient model_name & ImageCoord"""
    model_name: str
    image_coords: ImageCoord

   
class BaseModelList(BaseModel):
    """
    Contient deux listes :
    - num_features : List[NumFeatures]
    - image_coords : List[ImageCoord]
    """

    num_features: List[NumFeatures]
    image_coords: List[ImageCoord]

    # Pydantic 2 utilise model_config au lieu de Config
    model_config = {
        "arbitrary_types_allowed": True  # si NumFeatures ou ImageCoord sont des types personnalisés
    }

# Modèle Pydantic pour la liste des features
class PredictionRequest(BaseModel):
    model_name: str
    list_features: BaseModelList


# PRODUCTION-MODEL ENDPOISYNCHRONENT #######
def s_production_model():
    # Création du client mlflow
    client = mlflow.MlflowClient()
    # Récupération du modèle standard pour le CNN
    models = client.search_registered_models()
    # Exception levée si non trouvé
    if not models:
        raise ValueError("Modèle non trouvé.")
    for m in models:
        if "production_cnn" in m.aliases.keys():
            model = m
            version = m.aliases["production_cnn"]
            alias = "production_cnn"
    
    if not model:
        raise HTTPException(status_code=400, detail="Aucun modèle en production trouvé.")
    # Récupéversion = model.versionrsions avec aliases
    return f"{model.name}/{version}"
#######

description = """
### API de prédiction par CNN pour les spectrogrammes de musique - Faite par Sandra, Cyril, John et Adrien.

* Modèle entraîné sur des échantillons du dataset GTZAN ;
* Summary et Print disponible.

#### La documentation est disponible ci-dessous
"""

# Méta-données des sections #######
tags_metadata = [
    {   "name": "Informations",
        "descriprion": "Informations générales sur l'API"
    },

    {   "name": "Prédictions",
        "description": "Le cnn fera une prédiction sur le style de musique à partir de deux spectrogrammes fournis, un harmonique et l'autre percussif."
    }
]
#######

app = FastAPI(
    title="🪐 Music Classification model predict API",
    description=description,
    version="0.1",
    contact={
        "name": "Jedha",
        "url": "https://jedha.co",
    },
    openapi_tags=tags_metadata
)

"""
"/" : Message de bienvenue

"/production-model : retourne le modèle de production et sa version poour information

"/predict-cnn : retourne une prédiction sur une paire d'image fournie (spectrogramme harmonique, spectrogramme percussif)

"""

# BASE ENDPOINT #######
@app.get("/", tags=["Informations"])
async def index():

    message = """
    Bienvenue sur l'API du modèle de deep-learning sur les spectrogrammes de musiques\n
    Utilisez cette API à bon escient é_è"""

    return message
#######

# PRODUCTION-MODEL ENDPOINT #######
@app.get("/production-model", tags=["Prédictions"])
async def production_model():
    # Création du client mlflow
    client = mlflow.MlflowClient()
    # Récupération du modèle standard pour le CNN
    models = client.search_registered_models()
    # Exception levée si non trouvé
    if not models:
        raise ValueError("Modèle non trouvé.")
    for m in models:
        if "production_cnn" in m.aliases.keys():
            model = m
            version = m.aliases["production_cnn"]
            alias = "production_cnn"
    
    if not model:
        raise HTTPException(status_code=400, detail="Aucun modèle en production trouvé.")
    # Récupéversion = model.versionrsions avec aliases
    return f"{model.name}/{version}"
#######

# PRINT-MODEL ENDPOINT #######
@app.get("/print-model", tags=["Informations"])
async def print_modele():
    return {"model": str(CNN_MODEL)}
#######

# SUMMARY-MODEL ENDPOINT #######
@app.get("/summary-model", tags=["Informations"])
async def summary_modele():
    f = io.StringIO()
    with redirect_stdout(f):
        summary(CNN_MODEL, input_size = (1, 2, 128, 256))
    model_summary = f.getvalue()
    return {"summary" : model_summary}
#######

"""@app.get("/model_list_mlflow")
async def index():
    get the list of models of MLflow
    
    liste_models = list_mlflow_models(MLFLOW_TRACKING_URI)
    
    return liste_models

@app.get("/get-model")
async def get_model():
    return {"current_model_dict": model_dict}"""

# UPDATE-MODEL ENDPOINT
@app.get("/update-model")
async def update_model(payload: ItemModel):
    # On met à jour le modèle cnn avec la dernière version 'production-cnn'
    MODEL_ID = await production_model()
    CNN_MODEL = mlflow.pytorch.load_model("models:/" + f"{MODEL_ID}", map_location = "cpu")
    
    return "Modèle mis à jour avec succès"

    
"""@app.post("/predict_f")
async def predict_f(request: PredictionRequest_f):
    # Vérifier que le modèle demandé existe
    # if request.model_name not in model_dict:
    #     raise HTTPException(status_code=404, detail="Model not found")

    model_name = request.model_name
    num_features_0 = request.num_features
    
    num_features = pd.DataFrame(num_features_0)
    num_features = num_features.T
    num_features.columns = num_features.iloc[0]
    num_features = num_features.iloc[1:,:]
    print(num_features.columns)

    num_features.head()    
    
    print(type(num_features))
    model_type =  detect_model_type(model_name)
           
    # model_info = model_dict[request.model_key]
    # model_type = model_info["type"]
    # model_uri = model_info["model_uri"]

    # Charger le modèle MLflow
    try:
        model_uri = get_model_uri_hard(model_name,stage=MODEL_STAGE)
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # Sélectionner les features selon le type
    try:
        if model_type == "feature":
            print('model type =features')
            features_df = num_features
            features_df.head()
            # Colonnes à supprimer si présentes
            cols_to_drop = ["filename", "length","label"]
            
            features_df = features_df.drop(columns=[col for col in cols_to_drop if col in features_df.columns])
            print(features_df.head())
            # Faire la prédiction sur tout le DataFrame
            
            prediction = model.predict(features_df)
            print(f"prediction: {prediction}")
            # Retourner la liste complète des prédictions
            response = {"predictions": prediction.tolist()}
    
        # utiliser @app.post("/extract", tags=["features"]) pour fabriquer le json  
            
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
    except IndexError:
        raise HTTPException(status_code=400, detail="Insufficient features provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction}

"""
mytransform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean = [0.5], std = [0.5])])

@app.post("/predict-cnn")
async def predict_cnn(request:Request):
    try:
        spectro_data = await request.json()
        harmo_b64 = spectro_data["harmo_file"]
        percu_b64 = spectro_data["percu_file"]

        h_bytes = base64.b64decode(harmo_b64)
        p_bytes = base64.b64decode(percu_b64)

        img_h = Image.open(io.BytesIO(h_bytes)).convert("L")
        img_p = Image.open(io.BytesIO(p_bytes)).convert("L")

        tensor_h = mytransform(img_h).unsqueeze(0)
        tensor_p = mytransform(img_p).unsqueeze(0)

        input_tensor = torch.cat([tensor_h, tensor_p], dim = 1)

        CNN_MODEL.eval()

        with torch.no_grad():
            
            output = CNN_MODEL(input_tensor)
            prediction = torch.argmax(output, dim = 1).item()

        return {"prediction" : prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


MODEL_ID = s_production_model()
CNN_MODEL = mlflow.pytorch.load_model("models:/" + f"{MODEL_ID}", map_location = "cpu")