# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:42:05 2021

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
#Universal Sentence Encoder  

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
print("module %s loaded" % module_url)

sentence_embeddings = model(sentences)
query = "I had pizza and pasta"
query_vec = model([query])[0]

for sent in sentences:
  sim = cosine(query_vec, model([sent])[0])
  print("Sentence = ", sent, "; similarity = ", sim)
  
###############################################################################s
