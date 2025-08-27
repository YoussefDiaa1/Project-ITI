# Project-ITI

📝 Sentiment Analysis Project
📌 Overview

This project focuses on Sentiment Analysis for both English and Arabic text datasets.
We applied Machine Learning, Deep Learning (LSTM), and Transformer-based models to classify sentences into Positive or Negative sentiments.
Our best-performing model is BERT using Hugging Face's AutoModelForSequenceClassification, achieving 91% accuracy.

📂 Dataset

English Dataset → Cleaned & processed separately

Arabic Dataset → Cleaned & processed separately

Combined Dataset → English + Arabic for Deep Learning


⚙️ Preprocessing Steps

Text Cleaning

Lowercasing

Removing punctuation, numbers, and special characters

Removing stopwords

Expand Contractions for english data

Tokenization (Word-level & Subword for BERT)

Lemmatization

Vectorization

TF-IDF for ML models

Word Embeddings for LSTM

BERT Tokenizer for Transformers

🤖 Models Used
1️⃣ Machine Learning Models (English + Arabic)

Logistic Regression

Naive Bayes

Random Forest

Best ML Accuracy → 88% for English 

Best ML Accuracy → 85% for Arabic


2️⃣ Deep Learning Model (LSTM)

We implemented a Long Short-Term Memory (LSTM) model before moving to Transformers.

Architecture:

Embedding Layer

LSTM Layer

Dense Layers + Dropout

Optimizer: Adam

Loss: Binary Crossentropy

Activation: Sigmoid

LSTM Accuracy → 89%

3️⃣ Transformer Model (BERT) ✅ Best Model

We used Hugging Face's AutoModelForSequenceClassification with a BERT-base architecture.

Tokenizer → BertTokenizerFast

Model → AutoModelForSequenceClassification

Optimizer → AdamW

Loss → CrossEntropyLoss

Training Epochs → 4

Batch Size → 32

Learning Rate → 2e-5

BERT Model Accuracy → 91% ✅

📊 Results
Model	Language	Accuracy
Logistic Regression	English	88%
Naive Bayes	English	84%
Naive Bayes	Arabic	82%
Logistic Regression	arabic	85%
LSTM	Mixed	89%
BERT (Transformers)	Mixed	91% ✅

👥 Contributors

Youssef Diaa

Ahmed Nageh

Alaa Faisal

Walaa Magdy

Amr Hassan 

Youmna Ayman

📌 Future Work

Use AraBERT for better Arabic sentiment accuracy

Deploy a real-time prediction dashboard with Streamlit

Extend to multi-class sentiment analysis (positive, neutral, negative)
