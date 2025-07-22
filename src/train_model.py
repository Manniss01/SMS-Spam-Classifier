import pandas as pd
import pickle
import os
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.preprocess import clean_text, count_punct

nltk.download('stopwords')

# Load and preprocess data
data = pd.read_csv('../data/SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'body_text']
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(count_punct)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data[['body_text', 'body_len', 'punct%']], data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(analyzer=clean_text)
tfidf_train = tfidf.fit_transform(X_train['body_text'])

X_train_vect = pd.concat(
    [X_train[['body_len', 'punct%']].reset_index(drop=True),
     pd.DataFrame(tfidf_train.toarray())],
    axis=1
)

# Ensure all column names are strings
X_train_vect.columns = X_train_vect.columns.astype(str)

# Train model
clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train_vect, y_train)

# Save model and vectorizer
os.makedirs('../models', exist_ok=True)
with open('../models/rf_model.pkl', 'wb') as m, open('../models/tfidf_vectorizer.pkl', 'wb') as v:
    pickle.dump(clf, m)
    pickle.dump(tfidf, v)
