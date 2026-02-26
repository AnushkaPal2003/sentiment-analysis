import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df["clean_review"] = df["review"].apply(clean_text)
    return df[["clean_review", "label"]]