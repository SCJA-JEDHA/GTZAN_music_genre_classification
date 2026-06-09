FROM python:3.12-slim

RUN apt-get update -y 
RUN apt-get install nano unzip curl -y


# Leverage layer caching: copy only reqs first
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY API_music_predict_image.py API_music_predict_image.py
#COPY .env .

# # Copy all local files to /home/user/app with "user" as owner of these files
# # Always use --chown=user when using HUGGINGFACE to avoid permission errors
# COPY --chown=user . $HOME/app

CMD ["bash","-lc","fastapi run API_music_predict_image.py --host 0.0.0.0 --port ${PORT}"]
