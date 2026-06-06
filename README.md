# 🔍 Phishing Email Detection Model

A Python-based machine learning project that trains a classifier to distinguish between malicious phishing attempts and safe, legitimate emails using Natural Language Processing (NLP). 

This project demonstrates how data science principles can be defensively deployed to identify social engineering patterns and block threats at scale.

---

## 🚀 Features

* **Textual Feature Extraction:** Converts raw email bodies and subject lines into numerical matrices using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
* **Supervised Classification:** Utilizes a Multinomial Naive Bayes classifier highly optimized for textual pattern evaluation.
* **Security Keyword Tracking:** Automatically analyzes text for structural urgency indicators commonly found in social engineering attacks (e.g., "urgent", "verify account", "bank").
* **Performance Evaluation Metrics:** Computes overall model accuracy alongside a structured Confusion Matrix to break down True Positives and error distributions.

---

## 🧠 How It Works

The machine learning pipeline processes data through five core stages:

1. **Dataset Ingestion:** Loads labeled raw text data explicitly categorized as Phishing (`1`) or Safe/Ham (`0`).
2. **Text Processing & Vectorization:** Applies a TF-IDF Vectorizer to calculate the relative statistical weights of words, transforming sentences into a mathematical coordinate space.
3. **Data Splitting:** Segregates data into a **Training Set** ($80\%$) to teach the model and a **Testing Set** ($20\%$) for blind evaluation.
4. **Model Training:** Maps patterns and vocabulary distributions unique to phishing language.
5. **Evaluation Matrix Mapping:** Tests the trained classifier against the test set and outputs performance metrics.

---
