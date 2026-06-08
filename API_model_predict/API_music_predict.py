# API model predict 


import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi import HTTPException , Query   
from enum import Enum
import pandas as pd
from pydantic import BaseModel
from typing import Literal, List, Union
from typing import Any, Dict
from typing import List, Union
from pydantic import BaseModel, field_validator

import mlflow
from dotenv import load_dotenv
import numpy as np
import os
# Load environment variables from the .env file (if present)
load_dotenv()

# import model : 
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
REGISTERED_MODEL_NAME = "MGC_features_SVM_baseline"

IMAGE_NX          = 432
IMAGE_NY          = 288 
IMAGE_N = IMAGE_NX * IMAGE_NY
tracking_uri = "https://cyrilbrg-mlflow-music.hf.space/"

description = """
API for Music type classification
This API sends feature vector 
and gets model prediction class for "genre" 
 
## Introduction Endpoints

Here are two endpoints you can try:
* `/`: **GET** request that display a simple default message.
* `/greetings`: **GET** request that display a "hello message"

## Blog Endpoints

Imagine this API deals with blog articles. With the following endpoints, you can retrieve and create blog posts 
* `/blog-articles/{blog_id}`: **GET** request that retrieve a blog article given a `blog_id` as `int`.
* `/create-blog-article`: POST request that creates a new article

## Machine Learning

This is a Machine Learning endpoint that predict salary given some years of experience. Here is the endpoint:

* `/predict` that accepts `floats`


Check out documentation below 👇 for more information on each endpoint. 
"""


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
    chroma_mean : Union[int, float]
    chroma_var : Union[int, float]
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
    label : Union[int, float]
    
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

description = """
This is your app description, written in markdown code 

# This is a title 

* This is a bullet point 
"""

tags_metadata = [
    {
        "name": "Name_1",
        "description": "LOREM IPSUM NEC."
    },

    {
        "name": "Name_2",
        "description": "LOREM IPSUM NEC."
    }
]


# app.py 
# .transform(features)
# model.predict()
# MGC_features_SVM_baseline @challenger


...
...
# stocker un dico en var env : 
# my_dict = {"key1": "value1", "key2": "value2"}
#os.environ["MY_DICT"] = json.dumps(my_dict)
# model_dict = json.loads(os.environ["MODEL_DICT"])
# model_dict = {
#     "model1":{
#         "name":"MGC_features_SVM_baseline",
#         "type":"feature",
#         "model_uri":"models:/registered_model1@production"
#     },
#     "model2":{
#         "name":"toto",
#         "type":"image",
#         "model_uri":"models:/registered_model2@production"
        
#     }
# }
 

def list_mlflow_models(tracking_uri: str) -> list[str]:
    """Récupère la liste des modèles enregistrés dans MLflow."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        models = [m.name for m in client.search_registered_models()]
        return models if models else ["(aucun modèle trouvé)"]
    except Exception as e:
        return [f"Erreur MLflow : {e}"]

def get_model_uri(model_name, stage="Production"):
    return f"models:/{model_name}@{stage}"

def detect_model_type(model_name: str) -> str:
    name = model_name.lower()
    if "cnn" in name:
        return "image"
    if "feature" in name:
        return "feature"
    return "unknown"

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

/unique-values - To be able to get unique values of a given column in the dataset

/groupby - To be able to group by a given column and chose an aggregation metric
           Front-End should be able to choose 
           which categorical column to use and which aggregation metric (sum, mean, max, min...)
           
/filter-by - To be able to filter by one or several, categories within your dataset
Front-End should be able to choose one column and one or several categories within that column

/quantile - To be able to retrieve the top x% or bottom x% values of a given column in the dataset

i.e the top 10% DailyRate or the lowest 5% DistanceFromHome
Front-End should be able to choose the percentage and to choose whether it's the top or low values"""



@app.get("/model_list_mlflow")
async def index():
    """get the list of models of MLflow
    """
    liste_models = list_mlflow_models(MLFLOW_TRACKING_URI)
    
    return liste_models

@app.get("/get-model")
async def get_model():
    return {"current_model_dict": model_dict}


@app.post("/update-model")
async def update_model(payload: ItemModel):
    global model_dict
    
    # On met à jour la variable globale avec les données reçues
    model_dict = payload.data
    
    return {"message": "Dictionnaire mis à jour avec succès", "stored_data": model_dict}

    
@app.post("/predict")
async def predict(request: PredictionRequest):
    # Vérifier que le modèle demandé existe
    # if request.model_name not in model_dict:
    #     raise HTTPException(status_code=404, detail="Model not found")

    model_name = request.model_name
    list_features = request.list_features

    model_type =  detect_model_type(model_name)
           
    # model_info = model_dict[request.model_key]
    # model_type = model_info["type"]
    # model_uri = model_info["model_uri"]

    # Charger le modèle MLflow
    try:
        model_uri = get_model_uri(model_name,stage="challenger")
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # Sélectionner les features selon le type
    try:
        if model_type == "feature":
            # # On prend la première feature dans la liste
            # features = request.list_features.num_features  # C’est une liste de NumFeatures
            # #features = request.list_features[0]
            # features_df = pd.DataFrame([first_feature.dict()])
            # cols_to_drop = ["filename", "length","label"]
            # features_df = features_df.drop(columns=[col for col in cols_to_drop if col in features_df.columns])
            # # X = df.drop(columns=["filename","" "label"]) : 
            # # on enleve les colonnes 1,2 (filename,length) et la derniere colonne (label) (peut-etre)
            # # Suppression par noms de colonnes récupérés via leur position
            # cols_to_drop = [features.columns[0], features.columns[1], features.columns[-1]]
            # features_modified = features.drop(columns=cols_to_drop)
            
            # # La prédiction attend probablement un DataFrame ou un tableau 2D
            # # Adapter selon le modèle
            # prediction = model.predict([features])
            
            features_dicts = [feature.dict() for feature in request.list_features.num_features]

            # Créer un DataFrame pandas avec toutes les features
            features_df = pd.DataFrame(features_dicts)

            # Colonnes à supprimer si présentes
            cols_to_drop = ["filename", "length","label"]
            features_df = features_df.drop(columns=[col for col in cols_to_drop if col in features_df.columns])

            # Faire la prédiction sur tout le DataFrame
            prediction = model.predict(features_df)

            # Retourner la liste complète des prédictions
            response = {"predictions": prediction.tolist()}
    
        # utiliser @app.post("/extract", tags=["features"]) pour fabriquer le json  
            
        elif model_type == "image":
            # On prend la deuxième feature dans la liste
            #features_img = request.list_features[1]
            features_img = request.list_features.image_coords
            prediction = model.predict([features_img])
            
            response = {"prediction": prediction.tolist()}
        
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
    except IndexError:
        raise HTTPException(status_code=400, detail="Insufficient features provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction}
