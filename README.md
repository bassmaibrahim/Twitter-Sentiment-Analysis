# Twitter Sentiment Analysis — TF-IDF + Logistic Regression & BERT

## 📌 Objective
The main objective of this project is to classify tweets into three sentiment categories — Positive, Negative, and Neutral — using both traditional Machine Learning (TF-IDF + Logistic Regression) and a deep learning approach (BERT).
The project demonstrates the end-to-end process from data cleaning, EDA (Exploratory Data Analysis), model training, evaluation, and model saving.

## 🎯 Aims
- Perform sentiment classification on tweets mentioning various entities.
- Compare the performance of a traditional ML model (TF-IDF + Logistic Regression) and a transformer-based model (BERT).
- Preprocess and clean text to remove noise such as URLs, mentions, and special characters.
- Visualize sentiment distribution and frequent words per sentiment.
- Save trained models for future deployment.

## 📂 Dataset Used
Source: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

**Files/**
- `twitter_training.csv` — Training data **(55,328 rows × 4 columns)**
- `twitter_validation.csv` — Validation data **(3,799 rows × 4 columns)**
  
## 🛠 Tech Stack
### Languages: Python
### Libraries & Frameworks:
- Data Handling & Analysis: pandas, numpy
- Visualization: matplotlib, seaborn, WordCloud
- Machine Learning: scikit-learn (TfidfVectorizer, LogisticRegression)
- Deep Learning: TensorFlow, Transformers (BERT)
- Utilities: joblib, zipfile, shutil
### Models:
- TF-IDF + Logistic Regression (Baseline)
- BERT (bert-base-uncased) for sequence classification

## 📋 Project Overview & Process
- 1️⃣ Data Loading & Preparation
- 2️⃣ Data Cleaning
- 3️⃣ Exploratory Data Analysis (EDA)
- 4️⃣ Model 1: TF-IDF + Logistic Regression
- 5️⃣ Model 2: BERT
- 6️⃣ Model Comparison
- 7️⃣ Random Predictions
- 8️⃣ Model Saving
