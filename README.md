# Quora Insincere Questions Classification

An NLP machine learning project to detect insincere and toxic questions on Quora. Based on the [Kaggle competition](https://www.kaggle.com/c/quora-insincere-questions-classification).

## Problem Statement

Quora is a platform that empowers people to learn from each other. However, an existential problem is toxic and insincere content — questions that are worded in a non-neutral way to spread negativity or contain no real intent to get answers. This project builds a model to flag such questions automatically.

## Approach

- Text preprocessing (tokenisation, stopword removal, lemmatisation)
- Feature extraction using word embeddings (GloVe / Word2Vec)
- Classification using machine learning / deep learning models
- Evaluation using F1-score (as per Kaggle metric)

## Dataset

The dataset is sourced from the Kaggle competition and contains:
- `train.csv` — labelled training questions
- `test.csv` — unlabelled test questions
- `sample_submission.csv` — submission format

> Download the dataset from [Kaggle](https://www.kaggle.com/c/quora-insincere-questions-classification/data)

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn nltk keras tensorflow
```

### Run

```bash
jupyter notebook
```

Open the notebook and run cells sequentially.

## Results

| Model | F1 Score |
|-------|----------|
| Logistic Regression (baseline) | ~0.65 |
| LSTM with GloVe embeddings | ~0.70+ |

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, NLTK, Keras/TensorFlow
- **Notebook:** Jupyter
