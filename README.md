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
├── model.py                # Script for training and saving the ML model and vectorizer
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
│   └── predict.py              # Mode


