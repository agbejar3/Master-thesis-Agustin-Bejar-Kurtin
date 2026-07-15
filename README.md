# Uncertainty-Driven Cascade Models for Financial Sentiment Analysis

*Master's thesis project — University of Amsterdam, 2026*

## Overview

This project tests whether an uncertainty-driven cascade can offer a better balance between accuracy and computational cost than a single expensive model in financial text classification, while remaining interpretable. A lightweight logistic regression (Loughran-McDonald lexicon features + TF-IDF) handles sentences it is confident about; a fine-tuned FinBERT model is invoked only on the sentences the cheap stage declines to label. The pipeline is evaluated on Financial PhraseBank and SEntFiN, and on SEntFiN under two Stage 2 conditions — zero-shot transfer and in-domain fine-tuning — to test whether the cascade's value depends on the expensive stage actually being the stronger model.

## Repository Structure
'''
├── base_model.ipynb                  # Cascade pipeline on Financial PhraseBank (run this to reproduce PhraseBank results)
├── SentFin.ipynb                     # Cascade pipeline on SEntFiN, both Stage 2 conditions
├── finbert-finetuned/                 # FinBERT (finbert-tone) fine-tuned on SEntFiN's training split
├── financial_phrasebank_allagree.csv  # Dataset
└── SEntFiN.csv                        # Dataset
'''
> Note: exploratory notebooks, raw text extracts, and result plots (F1 benchmarks, confusion matrices, SHAP, reliability, threshold tradeoffs) are generated locally by running `base_model.ipynb` and `SentFin.ipynb`, and are not tracked in this repo — see `.gitignore`.

## Setup

```bash
pip install -r requirements.txt
```

> Generate `requirements.txt` from your working environment with:
> `pip freeze > requirements.txt`

## Data

- **Financial PhraseBank** (`financial_phrasebank_allagree.csv`) — Malo, P., Sinha, A., Korhonen, A., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.
- **SEntFiN** (`SEntFiN.csv`) — Sinha, A., Kedas, S., Kumar, R., & Malo, P. (2022). SEntFiN 1.0: Entity-aware sentiment analysis for financial news. *Journal of the Association for Information Science and Technology*, 73(9), 1314–1335.

> Note: verify each dataset's license before treating it as freely redistributable, even in a private repo.

## Usage

Run `base_model.ipynb` to reproduce the Financial PhraseBank results: Stage 1 training, threshold selection, Platt calibration, SHAP attribution, and the full system benchmark. Run `SentFin.ipynb` to reproduce the SEntFiN results under both Stage 2 conditions (zero-shot transfer from the PhraseBank checkpoint, and fine-tuned on SEntFiN directly using the checkpoint in `finbert-finetuned/`).

## Results

Running `base_model.ipynb` and `SentFin.ipynb` reproduces the key results locally: F1 benchmarks, confusion matrices, SHAP-based interpretability, and reliability/threshold tradeoff analysis. On Financial PhraseBank, the cascade reaches 0.951 accuracy against BERT's 0.965 ceiling while sending only 39.7% of sentences to Stage 2. On SEntFiN, the cascade matches BERT's accuracy exactly (0.870) once Stage 2 is fine-tuned in-domain, while calling it on only 66.2% of headlines. Plots are generated on the fly and are not stored in the repo.

## Author

Agustin Bejar Kurtin — University of Amsterdam
