# import important modules
import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB # classifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    plot_confusion_matrix,
)
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re #regular expression
# Download dependency
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
    "stopwords",
    "punkt",
    "omw-1.4"
):
    nltk.download(dependency)
    
import warnings
warnings.filterwarnings("ignore")

import joblib
import uvicorn
from fastapi import FastAPI

# seeding
np.random.seed(123)

df = pd.read_csv("final.csv")
print(df.shape)

# Dropping all columns except Class and Tweet. 

# Class: 

# 2 -> Neither

# 1 -> Offensive Language

# 0 -> hate_speech

#df = df.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'], axis=1)

stop_words =  stopwords.words('english')
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r'@[A-Za-z0-9]+','',text) # Removing @mentions
    text = re.sub(r'#','',text) # Removing the '#' symbol
    text = re.sub(r'RT[\s]+','',text) # Removing RT
    text = re.sub(r'https?:\/\/\S+','',text) # Removing hyperlinks
    text = re.sub(r'[^a-zA-Z ]',' ', text) # Removing all the punctuations and numbers
    text = text.lower()
        
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    words = word_tokenize(text)
    filtered_sentence = [w for w in words if not w in stop_words]
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)

df["cleaned_tweet"] = df["text"].apply(text_cleaning)

X = df["cleaned_tweet"]
y = df['label'].values

print(df.head())
print(y)

# split data into train and validate
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y,
)

sentiment_classifier = Pipeline(steps=[
                               ('pre_processing',TfidfVectorizer(lowercase=False)),
                                 ('logistic_regression', LogisticRegression(penalty = 'elasticnet', warm_start = True, max_iter = 1000,  C=1.3, solver='saga', l1_ratio=0.9))
                                 ])

sentiment_classifier.fit(X_train,y_train)

y_preds = sentiment_classifier.predict(X_valid)

print(accuracy_score(y_valid, y_preds))

joblib.dump(sentiment_classifier, 'sentiment_model_pipeline.pkl')

