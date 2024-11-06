---
title: Word Predictor
emoji: 🐢
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
license: mit
short_description: AI Masked Word Prediction Tool
---

# 📝 Masked Word Prediction Tool

A web tool that predicts missing words in sentences, using BERT to handle multiple masked words at once. Simply type your sentence with masked words (as `_`) and get accurate predictions in real-time.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Transformers](https://img.shields.io/badge/Transformers-FFB7B2?style=for-the-badge&logo=transformers&logoColor=black)](https://huggingface.co/transformers/)
[![Gradio](https://img.shields.io/badge/Gradio-4F9E5A?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FF3C00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Keras](https://img.shields.io/badge/Keras-FF3C00?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)



## 🎯 Features

- Predicts single or multiple masked words in a sentence.
- Fast predictions (0.3 to 1.8 seconds) using optimized BERT-based masked language modeling.
- User-friendly interface with customizable masked word token (`_` by default).

## 🚀 How It Works

The app leverages **BERT** for masked language modeling, using Hugging Face's `transformers` library for powerful NLP capabilities. Here’s how it handles predictions:

1. Replaces any `_` in the input text with the BERT `[MASK]` token.
2. Processes each `[MASK]` position in the sentence individually to predict likely words.

## 🔥 Try It Out

[Launch the App on Hugging Face Spaces](https://huggingface.co/spaces/tris-dev/word-predictor)

### Example Usage

Type in a sentence with `_` for masked words:

> _ order to achieve our goals, we need to focus on _ strategies that will improve our team’s _ and use our resources _.

The model might predict:

> **In** order to achieve our goals, we need to focus on **developing** strategies that will improve our team’s **performance** and use our resources **effectively**.

## ⚡ Performance

Thanks to optimization efforts, the app responds quickly, averaging between **0.3** to **1.8 seconds** per prediction depending on the input length.

## 🛠️ Installation (For Local Use)

1. Clone the repository and install dependencies to get started locally.
```bash
git clone https://github.com/trisDeveloper/AI-Word-Predictor
cd AI-Word-Predictor
pip install -r requirements.txt
```
2. Run the app with: `python app.py`

### 📄 License

MIT License