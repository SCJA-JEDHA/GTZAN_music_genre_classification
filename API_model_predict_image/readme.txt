docker build -t mon_api_modelpredict .
docker run --env-file .env -p 8000:8000 mon_api_modelpredict

# ou 
docker-compose up --build