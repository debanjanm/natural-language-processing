# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:59:58 2021

@author: acer
"""

###############################################################################
###############################################################################
#Hashing Vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

sentences = [
    'He is playing in the field.',
    'He is running towards the football.',
    'The football game ended.',
    'It started raining while everyone was playing in the field.'
]

vectorizer = HashingVectorizer(norm = None, n_features = 17)
sentence_vectors = vectorizer.fit_transform(sentences)
print(sentence_vectors.toarray())

###############################################################################
###############################################################################