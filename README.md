# 📩 SMS Spam Classifier

An intelligent and lightweight web application that classifies SMS messages as **Spam** or **Not Spam** using a robust machine learning pipeline. This project leverages **Natural Language Processing (NLP)** techniques, a **Random Forest Classifier**, **TF-IDF vectorization**, and handcrafted textual features. It is packaged with a responsive **Gradio interface** for interactive usage and easily deployable on platforms like Hugging Face Spaces.

---

## 📌 Key Features

- ✅ **Accurate SMS Spam Detection**  
  Classifies messages with high accuracy using a trained Random Forest model.

- 🧠 **TF-IDF + Metadata Features**  
  Uses Term Frequency-Inverse Document Frequency (TF-IDF) along with handcrafted features such as body length and punctuation ratio.

- ⚙️ **Modular Python Codebase**  
  Clean separation between preprocessing, model training, and prediction logic.

- 🌐 **Interactive Gradio UI**  
  Allows users to test predictions in real-time through a friendly web interface.

- 🚀 **Hugging Face Spaces Ready**  
  Fully compatible for deployment on Hugging Face using the Gradio SDK.

- 📊 **Trainable Model Pipeline**  
  Easily retrain or fine-tune using `model.py` and the included dataset.

---

## 🗂️ Project Structure
``` bash
sms-spam-classifier/
│
├── app.py                  # Entry point for the Gradio-based web application
├── requirements.txt        # List of Python dependencies
├── README.md               # Project documentation
│
├── data/                   # Dataset directory
│   └── SMSSpamCollection.tsv   # SMS Spam Collection dataset (labelled messages)
│
├── models/                 # Serialized machine learning artifacts
│   ├── rf_model.pkl            # Trained Random Forest classifier
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│
├── src/                    # Source code modules
│   ├── __init__.py             # Makes src a Python package
│   ├── preprocess.py           # Text cleaning and feature extraction utilities
│   ├── predict.py              # Mode
├   └── train-model.py                # Script for training and saving the ML model and vectorizer
```

---
## 🧠 Model Overview

### 📚 Dataset

We use the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), a well-known labeled dataset of 5,574 messages tagged as:
- `ham` – legitimate (non-spam) messages
- `spam` – promotional or scam content

Each line contains a label and the corresponding message text.

---

### 🧪 Preprocessing

Messages are processed via:

- Lowercasing text
- Removing punctuation and symbols
- Tokenization using `nltk.word_tokenize`
- Stopword removal using `nltk.corpus.stopwords`
- Stemming via `nltk.PorterStemmer`

---

### ✨ Feature Engineering

We combine **statistical** and **textual** features:

| Feature Name   | Description                                 |
|----------------|---------------------------------------------|
| `body_len`     | Length of the message (excluding spaces)     |
| `punct%`       | % of punctuation characters in the message   |
| `tfidf`        | Vectorized word features using TF-IDF        |

---

### 🤖 Model Architecture

**Algorithm:** `RandomForestClassifier`  
**Why?**
- Handles class imbalance well
- Robust to overfitting
- Works well with mixed numeric and sparse features

**Training Configuration:**
- `n_estimators=150`
- `random_state=42`
- Trained on 80% training, 20% test split

**Artifacts Generated:**
- `models/rf_model.pkl`
- `models/tfidf_vectorizer.pkl`

---

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Manniss01/SMS-Spam-Classifier
   cd sms-spam-classifier
   ```
   ![Gradio App](https://raw.githubusercontent.com/your-username/sms-spam-classifier/main/demo.png)




