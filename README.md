
# Sentiment Analysis for Vietnamese Product Reviews

## Overview

This project classifies Vietnamese product reviews into 3 categories: **severe defects (0)**, **minor issues (1)**, and **no issues (2)**. It starts with 6 initial labels, merges them into 3, augments the "severe defects" class, trains an LSTM model with Word2Vec embeddings, and visualizes Loss/Accuracy using Matplotlib. Designed for Kaggle.

## Features

- Merges 6 labels into 3 for classification.
- Augments data for "severe defects" (label 0).
- Uses Word2Vec and LSTM for sentiment analysis.
- Plots Loss/Accuracy over epochs.

## Requirements

- Python 3.7+
- Libraries: `pandas`, `torch`, `torchtext`, `underthesea`, `tqdm`, `numpy`, `scikit-learn`, `matplotlib`
- Input: `final_data.xlsx` (6 labels), `word2vec_vi_words_100dims.txt`

## Installation

1. Upload files to a Kaggle notebook.
2. Install dependencies:

   ```bash
   !pip install pandas torch torchtext underthesea tqdm numpy scikit-learn matplotlib
   ```
3. Ensure `final_data.xlsx` and `word2vec_vi_words_100dims.txt` are in `/kaggle/input/`.

## Data Processing

1. **Preprocessing**: Load `final_data.xlsx` (6 labels), fill missing `cleaned_data`, merge into 3 classes (0: severe defects, 1: minor issues, 2: no issues).
2. **Augmentation**: Apply synonym replacement for label 0, save as `augmented_data.xlsx`.
3. **Word2Vec**: Clean `word2vec_vi_words_100dims.txt` into `vi_word2vec.txt`.
4. **Dataset Prep**: Tokenize with `underthesea`, build vocabulary (`min_freq=3`), convert to tensors, split 80/20 (train/test).

## Training & Evaluation

- **Model**: LSTM (2 layers, 256 hidden units, unidirectional, 0.2 dropout), Word2Vec embeddings (100D), Focal Loss.
- **Optimization**: Batch size 64, learning rate 5e-3, `ReduceLROnPlateau`, early stopping (patience=3), 10 epochs.
- **Visualization**: Plot Loss/Accuracy, saved as `training_metrics.png`.
- **Evaluation**: Classification report on test set.

## Results

- Test accuracy &gt;70% .

## License

MIT License - see LICENSE file.

## Acknowledgments

- Word2Vec embeddings from pre-trained Vietnamese models.
- Thanks to PyTorch and Underthesea communities.
