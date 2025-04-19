
#  Chatbot RAG pour PDF

Ce projet propose deux versions d’un assistant intelligent permettant d’interroger un document PDF à l’aide d’un modèle de langage local et d’un système RAG (Retrieval-Augmented Generation).

## Deux versions disponibles

- `app.py` et `rag_pipeline.py` : version **simple**, qui traite **une seule question** à la fois sans historique de conversation.
- `appChat.py` et `rag_pipelineChat.py` : version **conversationnelle**, avec **mémoire contextuelle** permettant une interaction multi-tours avec le chatbot.

## Installation

Avant de commencer, installez les dépendances nécessaires avec la commande suivante :

```bash
pip install -r requirements.txt
```

Adaptez le fichier requirements.txt selon vos besoins (GPU, modèles quantisés, etc.).

## ▶️ Utilisation de la version conversationnelle

1. Lancez le script principal :

```bash
python appChat.py
```

2. Ouvrez votre navigateur à l'adresse : [http://localhost:7860/?](http://localhost:7860/?)

3. **Uploadez un fichier PDF**.

4. Patientez environ **30 à 60 secondes** pendant le traitement.

5. Posez vos questions.

## 🤖 Modèle utilisé

- Par défaut : `meta-llama/Llama-3.2-1B-Instruct`
