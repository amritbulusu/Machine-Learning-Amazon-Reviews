#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:35:58 2017

"""

import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


train_data=pd.read_csv('train_dummy.csv').dropna()
test_data=pd.read_csv('test_dummy.csv').dropna()

#print("Train data size:",train_data.shape)
#print("Test data size:",test_data.shape)

#preprocessing
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 3. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 5. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

train_reviews_clean = [review_to_words(rev) for rev in train_data['review']]
test_reviews_clean= [review_to_words(rev) for rev in test_data['review']]
#print("Finished preprocessing of data".center(50,'-'))

vec=TfidfVectorizer(max_features=2500,min_df=2)
train_input=vec.fit_transform(train_reviews_clean).todense()
train_output=train_data['rating']

# we only call "transform", not "fit_transform" as we did for the training set.
test_input=vec.transform(test_reviews_clean).todense()
test_output=test_data['rating']
'''
print(train_input.shape)
print(train_output.shape)
print(test_input.shape)
print(test_output.shape)
'''
clf = MultinomialNB(alpha=10.0)
clf = clf.fit(train_input, train_output)


print("line 69 of code".center(40,'-'))
predicted_output = clf.predict(test_input)
print(predicted_output)
print(accuracy_score(test_output, predicted_output))
print(confusion_matrix(test_output, predicted_output))
scores = cross_val_score(clf, train_input, train_output, cv=10)
print(scores)
fpr, tpr, thresholds = roc_curve(test_output, predicted_output, pos_label=2)



plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_curve )
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()