# github actions 

##1 : workflow streamlit :

on push on /streamlit : 
  version : main or dev 

launch tests on streamlit 

if tests ok : 
.yaml :
name: Sync streamlit vers HF Space

on:
  push:
    branches: [main]
    paths:
      - 'streamlit/**'


  jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Push vers le Space HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: cyrilbrg
          SPACE_REPO: cyrilbrg/music_streamlit

        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"

          # Split subtree dans une branche temporaire pour ne pousser que streamlit/
          git subtree split --prefix=streamlit -b hf-sync-branch

          git remote add hf-space "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${SPACE_REPO}"
          git push hf-space hf-sync-branch:main --force




name: Sync streamlit vers HF Space

on:
  push:
    branches: [main, dev]
    paths:
      - 'streamlit/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Installer les dépendances
        run: |
          python -m pip install --upgrade pip
          pip install -r streamlit/requirements.txt
          pip install pytest

      - name: Lancer les tests
        run: pytest streamlit/ -v

  sync:
    needs: test
    if: success()
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Push vers le Space HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: cyrilbrg
          SPACE_REPO: cyrilbrg/music_streamlit
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"

          git subtree split --prefix=streamlit -b hf-sync-branch

          git remote add hf-space "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${SPACE_REPO}"
          git push hf-space hf-sync-branch:main --force