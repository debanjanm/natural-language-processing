# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:49:29 2021

@author: acer
"""

###############################################################################
###############################################################################

text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

###############################################################################
#1. Tokenization using Python’s split() function
''' Let’s start with the split() method as it is the most basic one. It returns a list of strings after breaking 
the given string by the specified separator. By default, split() breaks a string at each space. We can change the separator 
to anything. '''

#Word Tokenization
text.split()

'''This is similar to word tokenization. Here, we study the structure of sentences in the analysis.
 A sentence usually ends with a full stop (.), so we can use “.” as a separator to break the string '''

#Sentence Tokenization
text.split('. ') 

'''One major drawback of using Python’s split() method is that we can use only one separator at a time.
 Another thing to note – in word tokenization, split() did not consider punctuation as a separate token.'''
 
###############################################################################
#2. Tokenization using Regular Expressions (RegEx)
import re

'''The re.findall() function finds all the words that match the pattern passed on it and stores it in the list.
The “\w” represents “any word character” which usually means alphanumeric (letters, numbers) and 
underscore (_). ‘+’ means any number of times. 
So [\w’]+ signals that the code should find all the alphanumeric characters until any other character is encountered.'''

#Word Tokenization
tokens = re.findall("[\w']+", text)
tokens

'''To perform sentence tokenization, we can use the re.split() function. 
This will split the text into sentences by passing a pattern into it. '''

#Sentence Tokenization
sentences = re.compile('[.!?] ').split(text)
sentences

'''Here, we have an edge over the split() method as we can pass multiple separators at the same time. 
In the above code, we used the re.compile() function wherein we passed [.?!]. 
This means that sentences will split as soon as any of these characters are encountered.'''

###############################################################################
#3. Tokenization using NLTK

#Word Tokenization
from nltk.tokenize import word_tokenize 
word_tokenize(text)

#Sentence Tokenization
from nltk.tokenize import sent_tokenize 
sent_tokenize(text)

###############################################################################
#4. Tokenization using the spaCy library

from spacy.lang.en import English
nlp = English()      # Load English tokenizer, tagger, parser, NER and word vectors
          
#Word Tokenization
my_doc = nlp(text)  #  "nlp" Object is used to create documents with linguistic annotations.

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)
token_list


#Sentence Tokenization

# Add the component to the pipeline
nlp.add_pipe('sentencizer')

#  "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text)

# create list of sentence tokens
sents_list = []
for sent in doc.sents:
    sents_list.append(sent.text)
sents_list

###############################################################################
#5. Tokenization using Keras

from keras.preprocessing.text import text_to_word_sequence

#Word Tokenization
result = text_to_word_sequence(text)
result

###############################################################################
#6. Tokenization using Gensim

from gensim.utils import tokenize

#Word Tokenization
list(tokenize(text))

#Sentence Tokenization
from gensim.summarization.textcleaner import split_sentences
result = split_sentences(text)
result

###############################################################################
###############################################################################