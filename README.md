# TweetClassification
This repo is for a Supervised Classification Machine Learning project that attempts to classify tweets made by Donald Trump and Justin Trudeau. It is one of my first projects in Natural Language Processing, documnet classification.

The dataset is a corpus of tweets which were collected in November 2017. They are available in CSV format from https://www.kaggle.com/datasets/unofficialmerve/tweets-of-trump-and-trudeau 

Pre-processing included doing the usual cleaning with tweets (using lower case, removing URLs, @mentions, #hashtags and numeric data, as well as "rt" for retweet. It also included removing punctuation and whitespaces. 
Using DetectorFactory to detect the languages used in the tweets, I have found that out of the 400 tweets, 295 are in English, 95 are in French and the remaining 10 tweets are in other several languages (Italian, Chinese, etc.). I took the decision to remove these ten tweets as they are going to disrupt proper lemmatization.

I used Spacy's stopwords for both English and French, by looping on the dataset and using the proper set of stopwords accordig the language of the tweet as detected. 

I also used Spacy's lemmatizer according to the language of the tweet.

After that, I used wordcloud for visualization of the most frequently used lemmas by Trump and Trudeau.

For vectorizing, I used TfidfVectorizer, after splitting the data into training and testing datasets to avoid data leakage. The maximum number of features was set to 4000 after several attempts and I used the uni- and bi-grams.

For classification, I used Multinomial NaiveBayes, Logistic Regression and RandomForest. Logistic Regression scored the highest.

Libraries Used are:
streamlit
numpy
pandas
matplotlib
wordcloud
nltk
langdetect
textblob
scikit-learn
langdetect

Author: Radwa Kotait
