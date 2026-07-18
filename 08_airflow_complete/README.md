# 🚀 Introduction à Airflow & ETL

Repo de démonstration pour la journée **Intro Airflow + ETL** : installer Airflow proprement avec Docker Compose, comprendre les concepts clés (DAG, tâches, opérateurs, XCom), puis construire de vrais pipelines ETL — du plus simple (fichiers locaux) au plus réaliste (S3, PostgreSQL, MLflow).

## 🎯 Objectifs

1. Déployer Airflow en local avec Docker Compose
2. Comprendre l'anatomie d'un DAG et les dépendances entre tâches
3. Structurer un pipeline (parallélisme, TaskGroups, branching)
4. Construire un ETL complet avec XCom et un volume de données
5. Configurer des **Connexions** et **Variables** Airflow pour interagir avec des services externes (S3, PostgreSQL, MLflow)

## 🛠 Prérequis

- **Docker** et **Docker Compose** installés et lancés
- Au moins **4 Go de RAM** alloués à Docker (macOS : Docker Desktop → *Settings* → *Resources* → *Memory*)

Pour les DAGs `01` à `05` : aucun compte cloud nécessaire ✅

Pour le DAG `06` et l'opérateur custom :
- Un compte **AWS** avec un **bucket S3**
- Une base **PostgreSQL** accessible (NeonDB, RDS, ou un conteneur Docker)
- Un serveur **MLflow** avec un modèle enregistré (DAG `06` uniquement)

## 📁 Structure du repo

```
.
├── docker-compose.yaml   # Stack Airflow complète (CeleryExecutor + Postgres + Redis)
├── Dockerfile            # Image Airflow 2.10.4 (Python 3.11) + dépendances Python
├── requirements.txt      # pandas, providers Postgres/AWS, mlflow, scikit-learn...
├── dags/                 # Les DAGs, dans l'ordre pédagogique 👇
│   ├── 01_parallel_tasks.py          # Dépendances & parallélisme
│   ├── 02_taskgroups.py              # Organiser son graphe avec TaskGroup
│   ├── 03_conditional_branching.py   # BranchPythonOperator & trigger rules
│   ├── 04_etl_structure.py           # Squelette d'un ETL (design first!)
│   ├── 05_covid_etl.py               # ETL réel : data.gouv.fr → pandas → CSV
│   └── 06_mlflow_predict.py          # Prédiction batch depuis un registry MLflow
├── plugins/
│   └── s3_to_postgres.py             # Opérateur custom : S3 → PostgreSQL
├── data/                 # Sortie des pipelines (volume monté sur /opt/airflow/data)
└── logs/                 # Logs Airflow (volume monté)
```

## ⚡️ Démarrage

### 1. (Linux uniquement) Configurer l'UID
# specificité à linux 
```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

Sur macOS/Windows, cette étape est inutile.

### 2. Initialiser la base de données Airflow
# creer les 4 repertoires : 
# mkdir avec 4 arguments ne marche pas sous powershell ...
```bash
New-Item -ItemType Directory -Path .dags,.logs,.plugin,.data
```
À faire **une seule fois**, au premier lancement (l'image se build automatiquement, comptez quelques minutes) :

```bash
docker compose up airflow-init
```

Attendez le message `airflow-init exited with code 0`.

### 3. Lancer Airflow
# a la racine du dossier de lancement d'airflow 
mkdir ./dags ./logs ./plugin ./data
```bash
docker compose up
```

(ajoutez `-d` pour lancer en arrière-plan)

### 4. Accéder à l'interface

- Ouvrez [http://localhost:8080](http://localhost:8080)
- Identifiants par défaut : `airflow` / `airflow`

## 🧪 Utiliser les DAGs

Les DAGs sont **en pause à la création** (comportement volontaire, configuré dans le compose). Pour en exécuter un :

1. Activez-le avec le toggle à gauche de son nom
2. Déclenchez-le manuellement avec le bouton ▶️ (*Trigger DAG*)
3. Explorez les vues **Graph** et **Grid**, puis les **logs** de chaque tâche

Suivez l'ordre `01 → 06` : chaque DAG introduit un concept qui sert au suivant.

Pour le DAG `05_covid_etl`, les fichiers produits apparaissent dans le dossier `./data` de votre machine. Le DAG `06_mlflow_predict` nécessite la configuration des sections suivantes 👇

---

## 🔌 Connexions Airflow

Les **Connexions** stockent les credentials des services externes (bases de données, cloud...) de manière centralisée et chiffrée. Vos DAGs y font référence par leur `Conn Id` — **jamais de mot de passe en dur dans le code !**

Elles se configurent dans l'UI : **Admin → Connections → +**

### Connexion AWS (S3)

| Champ | Valeur |
| --- | --- |
| Conn Id | `aws_default` |
| Conn Type | `Amazon Web Services` |
| AWS Access Key ID | votre access key |
| AWS Secret Access Key | votre secret key |
| Extra | `{"region_name": "eu-west-3"}` (adaptez la région) |

💡 Créez de préférence un utilisateur IAM dédié avec des droits limités à votre bucket — inutile (et risqué) d'utiliser vos clés root.

### Connexion PostgreSQL

| Champ | Valeur |
| --- | --- |
| Conn Id | `postgres_default` |
| Conn Type | `Postgres` |
| Host | hôte de votre base (ex : `ep-xxx.eu-central-1.aws.neon.tech`) |
| Database | nom de la base |
| Login | utilisateur |
| Password | mot de passe |
| Port | `5432` |
| Extra | `{"sslmode": "require"}` (obligatoire pour NeonDB et la plupart des bases managées) |

⚠️ Il s'agit de **votre** base de données métier (NeonDB, RDS...), pas de la base Postgres interne du docker-compose — celle-ci est réservée aux métadonnées d'Airflow, on n'y écrit jamais ses propres données.

## 🔑 Variables Airflow

Les **Variables** stockent la configuration de vos DAGs (noms de buckets, URLs, clés d'API...). Elles se configurent dans **Admin → Variables → +**

Variables utilisées dans ce repo :

| Clé | Valeur | Utilisée par |
| --- | --- | --- |
| `S3BucketName` | nom de votre bucket S3 | opérateur `S3ToPostgresOperator` (templating) |
| `MLFLOW_TRACKING_URI` | URL de votre serveur MLflow (ex : `https://xxx.hf.space`) | DAG `06_mlflow_predict` |

Dans le code, on y accède de deux façons :

```python
# En Python, dans une fonction de tâche
from airflow.models import Variable
bucket = Variable.get("S3BucketName")

# En templating Jinja, dans les paramètres d'un opérateur
bucket = "{{ var.value.S3BucketName }}"
```

💡 Une Variable dont le nom contient `SECRET`, `PASSWORD`, `API_KEY`... est automatiquement masquée dans l'UI et les logs.

# variables env et connexions chargees avec un fichier : 
Pour ajouter automatiquement les variables d'environnement dans votre déploiement airflow, vous pouvez utiliser la solution suivante (qui rentre dans le tout début de votre docker-compose.yaml).
A noter qu'il faudra un .env au même endroit que votre docker-compose.yaml dans votre arborescence.

x-airflow-common:
  &airflow-common
  # In order to add custom dependencies or upgrade provider distributions you can use your extended image.
  # Comment the image line, place your Dockerfile in the directory where you placed the docker-compose.yaml
  # and uncomment the "build" line below, Then run `docker-compose build` to build the images.
  # image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:3.1.3}
  build: .
  env_file:
    - ${ENV_FILE_PATH:-.env}
  
  connexions : 
  dans le .env, les variables s'appellent forcement _CONN_
    Airflow lit les connexions depuis les variables d'environnement au format :
    AIRFLOW_CONN_{CONN_ID_EN_MAJUSCULES}=<URI>
    exemple : 
    # Connexion PostgreSQL
    AIRFLOW_CONN_MY_POSTGRES=postgresql://airflow_user:airflow_pass@localhost:5432/airflow_db

    # Connexion S3 / AWS
    AIRFLOW_CONN_AWS_DEFAULT=aws://AKIAxxxxxx:secretkeyxxxx@?region_name=eu-west-3

    # Connexion HTTP (API)
    AIRFLOW_CONN_MY_API=http://user:pass@api.example.com:443?endpoint=/v1


## 🧩 Plugin : opérateur custom `S3ToPostgresOperator`

Le dossier `plugins/` est monté dans les conteneurs et ajouté au `PYTHONPATH` : tout module qui s'y trouve est importable directement depuis vos DAGs.

`s3_to_postgres.py` montre comment créer son propre opérateur en héritant de `BaseOperator` :

- le constructeur reçoit les paramètres (`bucket`, `key`, `table`, connexions),
- `template_fields` rend ces paramètres compatibles avec le templating Jinja,
- `execute()` contient la logique : téléchargement S3 → lecture pandas → écriture Postgres via `to_sql`.

Utilisation dans un DAG :

```python
from s3_to_postgres import S3ToPostgresOperator

transfer = S3ToPostgresOperator(
    task_id="transfer_to_postgres",
    table="ma_table",
    bucket="{{ var.value.S3BucketName }}",
    key="mon_fichier.csv",
    postgres_conn_id="postgres_default",
    aws_conn_id="aws_default",
)
```

Prérequis : les connexions `aws_default` et `postgres_default` (section Connexions ☝️) et la variable `S3BucketName`.

## 🤖 DAG MLflow (`06_mlflow_predict`)

Ce DAG illustre un cas MLOps classique : **une prédiction batch orchestrée par Airflow**.

1. Il lit l'URL du serveur MLflow dans la variable `MLFLOW_TRACKING_URI`
2. Il récupère les credentials AWS depuis la connexion `aws_default` (nécessaires pour lire les artefacts du modèle sur S3)
3. Il charge le modèle depuis le **Model Registry** via son alias : `models:/ibm_attrition_detector@production`
4. Il télécharge un jeu de données et prédit sur un échantillon

Pour l'adapter à votre propre modèle, changez simplement `MODEL_URI` (format : `models:/<nom_du_modele>@<alias>`).

À noter dans le code : les imports lourds (`mlflow`, `pandas`, `boto3`) sont faits **dans la fonction** et non en haut du fichier — le scheduler parse tous les DAGs en continu, des imports lourds au niveau module ralentissent tout Airflow.

---

## 🔧 Commandes utiles

```bash
# Arrêter les conteneurs
docker compose down

# Tout arrêter ET supprimer la base (reset complet)
docker compose down --volumes --remove-orphans

# Rebuilder l'image après modification du requirements.txt
docker compose build

# Tester un DAG en CLI sans passer par l'UI
docker compose exec airflow-scheduler airflow dags test 05_covid_etl

# Lister les erreurs d'import de DAGs
docker compose exec airflow-scheduler airflow dags list-import-errors
```

## 🩺 Dépannage

| Symptôme | Cause probable | Solution |
| --- | --- | --- |
| Le webserver redémarre en boucle | RAM Docker insuffisante | Allouer ≥ 4 Go à Docker |
| `Permission denied` sur `logs/` ou `data/` (Linux) | UID non configuré | Créer le fichier `.env` (étape 1) |
| Un DAG n'apparaît pas dans l'UI | Erreur d'import Python | Bandeau rouge en haut de l'UI, ou `airflow dags list-import-errors` |
| `ModuleNotFoundError` dans une tâche | Dépendance absente de l'image | L'ajouter au `requirements.txt` puis `docker compose build` |
| Erreur SSL sur Postgres | `sslmode` manquant | Ajouter `{"sslmode": "require"}` dans l'Extra de la connexion |
| `NoCredentialsError` sur S3 | Connexion `aws_default` absente ou mal nommée | Vérifier le `Conn Id` exact dans Admin → Connections |
| Modifications d'un DAG invisibles | Cache du scheduler | Attendre ~30 s, ou rafraîchir la page |

## 📚 Pour aller plus loin

- [Documentation officielle Airflow](https://airflow.apache.org/docs/apache-airflow/stable/)
- [Gestion des Connexions](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html) et [des Variables](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/variables.html)
- [Créer un opérateur custom](https://airflow.apache.org/docs/apache-airflow/stable/howto/custom-operator.html)
- [Liste des opérateurs et providers](https://airflow.apache.org/docs/apache-airflow-providers/operators-and-hooks-ref/index.html)
- [Bonnes pratiques d'écriture de DAGs](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
