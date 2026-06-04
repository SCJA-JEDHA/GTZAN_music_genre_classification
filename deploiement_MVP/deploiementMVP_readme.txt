Deploiement : 
base de donnees NeonDB : 
postgresql://neondb_owner:npg_vmSGU6HT3yPp@ep-hidden-cake-agdphzbl-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

credentals aux autres 

bucket S3 :
============= 
music-classification-project2
    mlflow-disc : https://music-classification-project2.s3.eu-west-3.amazonaws.com/mlflow-disc/
        URI : s3://music-classification-project2/mlflow-disc/
        mlflow access key : AKIA5Q4LO2TJM66H25JC
        voir le secret access key ds fichier
  
  
    music-database : 
    URI : s3://music-classification-project2/music-database/
    les data sont sous : ./gtzan-dataset-music-genre-classification/Data 
    meme structure qu'en decompressant l'archive Zip

    il faut creer 4 credentals pour les 4 membres de l'equipe (pour qu'ils aient les droits de lire et stocker des trucs dessus)
    => 

    Pour l'API stocker les models_transformers dans ./models_transformers
    avec comme nom : nom_model_transformer.xxx
    comme ca, l'API predict pourra le recuperer
    