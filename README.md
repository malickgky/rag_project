# RAG Project — Setup & Guide

## Prérequis
- Python ≥ 3.11
- [UV](https://docs.astral.sh/uv/) installé : `pip install uv` ou `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Clé OpenAI dans un fichier `.env`

---

## 1. Initialisation avec UV

```bash
# Créer et initialiser le projet
uv init rag_project
cd rag_project

# Créer l'environnement virtuel Python 3.11
uv venv --python 3.11

# Activer l'environnement virtuel
# Windows :
.venv\Scripts\activate
# Linux / macOS :
source .venv/bin/activate

# Installer toutes les dépendances depuis pyproject.toml
uv sync
```

---

## 2. Variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
OPENAI_API_KEY=sk-...votre_clé...
```

---

## 3. Structure du projet

```
rag_project/
├── .env
├── pyproject.toml
├── README.md
├── assets/
│   └── rag.png            ← logo optionnel pour Streamlit
├── notebooks/
│   └── rag_notebook.ipynb ← Tests Indexing / Retrieval / Generation / Evaluation
└── src/
    ├── rag_chatbot.py     ← Partie 2 : Chatbot Streamlit (RAG classique)
    └── rag_multimodal.py  ← Partie 3 : RAG Multimodal (texte + images)
```

---

## 4. Lancer les applications

```bash
# Partie 2 — Chatbot RAG
streamlit run src/rag_chatbot.py

# Partie 3 — RAG Multimodal
streamlit run src/rag_multimodal.py

# Notebook (Partie 1)
jupyter notebook notebooks/rag_notebook.ipynb
```
