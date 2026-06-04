---
title: Mlflow
emoji: 🦀
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
short_description: mlflow demo
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
docker build -t mlflow-server .
docker run -p 4000:4000 mlflow-server