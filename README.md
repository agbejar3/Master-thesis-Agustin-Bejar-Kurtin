# FinBERT Fine-Tuning for Financial Sentiment Analysis

*Master's thesis project — [your university], [year]*

## Overview

[1–3 sentences: what research question you're answering, e.g. "This project fine-tunes FinBERT on financial sentiment datasets and evaluates its reliability, robustness, and interpretability compared to a baseline model."]

## Repository Structure

```
├── base_model.ipynb                  # Model training/evaluation (run this to reproduce everything)
├── finbert-finetuned/                 # Fine-tuned model code/config
├── financial_phrasebank_allagree.csv  # Dataset
└── SEntFiN.csv                        # Dataset
```

> Note: exploratory notebooks, raw text extracts, and result plots (F1 benchmarks, confusion matrices, SHAP, reliability, threshold tradeoffs) are generated locally by running `base_model.ipynb` and are not tracked in this repo — see `.gitignore`.

## Setup

```bash
pip install -r requirements.txt
```

> Generate `requirements.txt` from your working environment with:
> `pip freeze > requirements.txt`

## Data

- **Financial PhraseBank** (`financial_phrasebank_allagree.csv`, `Sentences_AllAgree.txt`) — [add citation/source link]
- **SEntFiN** (`SEntFiN.csv`) — [add citation/source link]

> Note: verify each dataset's license before treating it as freely redistributable, even in a private repo.

## Usage

[Brief instructions: e.g. "Run `Data_exploration.ipynb` first to inspect the data, then `base_model.ipynb` to reproduce baseline results."]

## Results

Running `base_model.ipynb` reproduces the key results locally: F1 benchmarks, confusion matrices, SHAP-based interpretability, and reliability/threshold tradeoff analysis. Plots are generated on the fly and are not stored in the repo.

## Author

[Your name] — [contact/email, optional]
