# TEXT PREPROCESSING 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
#import seaborn as sns
#color = sns.color_palette()
from sklearn.model_selection import train_test_split
import csv        
import re                        # csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

table = str.maketrans({key: None for key in string.punctuation})
def preProcess(text):
    breakpoint()
    # Should return a list of tokens
    lemmatizer = WordNetLemmatizer()
    breakpoint()
    filtered_tokens=[]
    lemmatized_tokens = []
    stop_words = set(stopwords.words('english'))
    text = text.translate(table)
    for w in text.split(" "):
        w = w.lower()
        if w not in stop_words and re.match( '^[a-z]+$', w):
            lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))
        filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens
    return filtered_tokens

featureDict = {} # A global dictionary of features

def toFeatureVector(tokens):
    localDict = {}
    for token in tokens:
        if token not in featureDict:
            featureDict[token] = 1
        else:
            featureDict[token] = +1
   
        if token not in localDict:
            localDict[token] = 1
        else:
            localDict[token] = +1
    
    return localDict


x = """ The coolest gloves. Pretty useful & comfortable too. 
        The customer service was very helpful. It wasn't clear if these are in men's or women's sizes. 
        I was well received & helped by a courteous & knowledgeable staff. 
        I'm very familiar with customer service as I manage a group of sales staff myself."""



temp  = preProcess( x)
print( temp)

dic  = toFeatureVector( temp)

print( featureDict)

#print(dic)



