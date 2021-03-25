# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:05:48 2021

@author: acer
"""

###############################################################################
###############################################################################
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
###############################################################################

sentences = ["I ate dinner.", 
       "We had a three-course meal.", 
       "Brad came to dinner with us.",
       "He loves fish tacos.",
       "In the end, we all felt like we ate too much.",
       "We all agreed; it was a magnificent evening."]


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

###############################################################################
###############################################################################
#SentenceBERT

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


sentence_embeddings = sbert_model.encode(sentences)

#print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
#print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])


query = "I had pizza and pasta"
query_vec = sbert_model.encode([query])[0]


for sent in sentences:
  sim = cosine(query_vec, sbert_model.encode([sent])[0])
  print("Sentence = ", sent, "; similarity = ", sim)