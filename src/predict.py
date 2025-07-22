import pickle
import pandas as pd
from src.preprocess import count_punct

# Load the model and vectorizer
with open('../models/rf_model.pkl', 'rb') as m, open('../models/tfidf_vectorizer.pkl', 'rb') as v:
    model = pickle.load(m)
    vectorizer = pickle.load(v)

def predict_spam(message):
    body_len = len(message) - message.count(" ")
    punct_percent = count_punct(message)

    tfidf_msg = vectorizer.transform([message])
    features = pd.DataFrame([[body_len, punct_percent]], columns=['body_len', 'punct%'])

    input_vect = pd.concat([
        features.reset_index(drop=True),
        pd.DataFrame(tfidf_msg.toarray())
    ], axis=1)

    # Convert all column names to strings to avoid TypeError in sklearn
    input_vect.columns = input_vect.columns.astype(str)

    return model.predict(input_vect)[0]
