# main libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import spacy
import re
import os
from langdetect import detect, DetectorFactory
from sklearn.model_selection import train_test_split

# Read the data
TRAIN_DATA_PATH = os.path.join(os.getcwd(), r'D:\Data Science Diploma\finalTest\test\tweets.csv')
df = pd.read_csv(TRAIN_DATA_PATH)
# Change values in author (label column) to "trump" and "trudeau"
df["author"] = df["author"].str.replace('Donald J. Trump', 'trump')
df["author"] = df["author"].str.replace('Justin Trudeau', 'trudeau')
df.drop(columns=["id"], axis=1, inplace=True)
# Change all tweets in status to lower case
def lowerCase(data):
    return data.lower()
df["clean_tweet"] = df["status"].apply(lambda x: lowerCase(x))

# clean URLs, hashtags, mentions and numeric data
def cleaning(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+)|(http?://[^s]+)|([^s]+.com)|(@[^\s]+)|(#)|(rt)|(-?\d+(\.\d+)?))', ' ', str(data))
df["clean_tweet"] = df["clean_tweet"].apply(lambda x: cleaning(x))

#remove "rt" from tweet
def remove_rt(data):
    return [str(data).replace("rt", "") if "rt" in data else data]
df["clean_tweet"] = df["clean_tweet"].apply(lambda x: remove_rt(x))

#Clean punctuation
def clean_punct(data):
    return re.sub(r'[^\w\s]+', "", str(data))
df["clean_tweet"] = df["clean_tweet"].apply(lambda x: clean_punct(x))

#Clean whitespace
def clean_whitespace (data):
    if isinstance(data, str):
        return " ".join(data.split())
    else:
        return data
df["clean_tweet"] = df["clean_tweet"].apply(lambda x: clean_whitespace(x))

# Loop over the sentences in the list and detect their language
languages = []
for sentence in df["clean_tweet"]:
    DetectorFactory.seed = 0
    languages.append(detect(sentence))
df["languages"] = languages
df= df[df["languages"].isin(['en', 'fr'])]

# Remove stopwords from English and French Tweets
en_core = spacy.load('en_core_web_sm')
fr_core = spacy.load("fr_core_news_sm")
# Function to remove stopwords based on language
def remove_stopwords(data, language):
    if language == "en":
        doc = en_core(data)
        stopwords = en_core.Defaults.stop_words
    elif language == "fr":
        doc = fr_core(data)
        stopwords = fr_core.Defaults.stop_words
    else:
        return data
    
    tokens = [token.text for token in doc if token.text.lower() not in stopwords]
    return ' '.join(tokens)

df["clean_tweet"] = df.apply(lambda row: remove_stopwords(row["clean_tweet"], row["languages"]), axis=1)

#lemmatizer
# Load the spaCy language models
en_core = spacy.load('en_core_web_sm', disable='parser')
fr_core = spacy.load("fr_core_news_sm", disable='parser')

# Function to lemmatize text based on language
def lemmatize_text(data, language):
    if language == "en":
        doc = en_core(data)
    elif language == "fr":
        doc = fr_core(data)
    else:
        return data
    
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

# Lemmatize the content in the specified column based on language
df['lemmatized'] = df.apply(lambda row: lemmatize_text(row['clean_tweet'], row["languages"]), axis=1)

#Drop the status column as the clean_tweet column replaces it
df.drop(columns=["status", "clean_tweet", "languages"], axis=1, inplace=True)
df.tail()

# Replace "trump" with 0 and "trudeau" with 1
df['author'] = df['author'].replace({'trump': 0, 'trudeau': 1})

# Splitting into features and target
X = df.drop(columns=['author'], axis=1)
y = df['author']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

#Initialize the TF-IDF Vectorizer
tvec= TfidfVectorizer(max_features= 4000, ngram_range= (1,2))
t_train=tvec.fit_transform(X_train["lemmatized"])
#t_test=tvec.transform(X_test["lemmatized"])

#define the pipeline that will preprocess the input tweet
def process_new(new_text, language):
    x = lowerCase(new_text)
    y = cleaning(x)
    z = remove_rt(y)
    a = clean_punct (z)
    b = clean_whitespace (a)
    c = remove_stopwords(b, language = language)
    final_format = lemmatize_text(c, language = language)

    vectorized_text = tvec.transform([final_format])
    
    return vectorized_text
