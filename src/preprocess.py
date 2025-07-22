import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = stopwords.words('english')

def clean_text(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    tokens = re.split('\W+', text)
    return [ps.stem(word) for word in tokens if word not in stop_words]

def count_punct(text):
    return round(sum([1 for char in text if char in string.punctuation]) / (len(text) - text.count(" ")), 3) * 100
