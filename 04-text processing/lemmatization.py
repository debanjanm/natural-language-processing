# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 19:06:27 2021

@author: acer
"""

###############################################################################
###############################################################################
#1. Wordnet Lemmatizer 
'''
Wordnet is a publicly available lexical database of over 200 languages that provides 
semantic relationships between its words. It is one of the earliest and most commonly used lemmatizer technique. 

It is present in the nltk library in python.
Wordnet links words into semantic relations. ( eg. synonyms )
It groups synonyms in the form of synsets.
synsets : a group of data elements that are semantically equivalent. 

'''

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()

# single word lemmatization examples
list1 = ['kites', 'babies', 'dogs', 'flying', 'smiling',
		'driving', 'died', 'tried', 'feet']
for words in list1:
	print(words + " ---> " + wnl.lemmatize(words))
	
#> kites ---> kite
#> babies ---> baby
#> dogs ---> dog
#> flying ---> flying
#> smiling ---> smiling
#> driving ---> driving
#> died ---> died
#> tried ---> tried
#> feet ---> foot

# sentence lemmatization examples
string = 'the cat is sitting with the bats on the striped mat under many flying geese'

# Converting String into tokens
list2 = nltk.word_tokenize(string)
print(list2)
#> ['the', 'cat', 'is', 'sitting', 'with', 'the', 'bats', 'on',
# 'the', 'striped', 'mat', 'under', 'many', 'flying', 'geese']

lemmatized_string = ' '.join([wnl.lemmatize(words) for words in list2])

print(lemmatized_string)
#> the cat is sitting with the bat on the striped mat under many flying goose

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""
               
               
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)      


###############################################################################
#2. Wordnet Lemmatizer (with POS tag)
'''
In the above approach, we observed that Wordnet results were not up to the mark. 
Words like ‘sitting’, ‘flying’ etc remained the same after lemmatization. 
This is because these words are treated as a noun in the given sentence rather than a verb. 
To overcome come this, we use POS (Part of Speech) tags. 

We add a tag with a particular word defining its type (verb, noun, adjective etc). 
For Example,

Word      +    Type (POS tag)     —>     Lemmatized Word
driving    +    verb      ‘v’            —>     drive
dogs       +    noun      ‘n’           —>     dog

'''

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# Define function to lemmatize each word with its POS tag

# POS_TAGGER_FUNCTION : TYPE 1
def pos_tagger(nltk_tag):
	if nltk_tag.startswith('J'):
		return wordnet.ADJ
	elif nltk_tag.startswith('V'):
		return wordnet.VERB
	elif nltk_tag.startswith('N'):
		return wordnet.NOUN
	elif nltk_tag.startswith('R'):
		return wordnet.ADV
	else:		
		return None

sentence = 'the cat is sitting with the bats on the striped mat under many badly flying geese'

# tokenize the sentence and find the POS tag for each token
pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

print(pos_tagged)
#>[('the', 'DT'), ('cat', 'NN'), ('is', 'VBZ'), ('sitting', 'VBG'), ('with', 'IN'),
# ('the', 'DT'), ('bats', 'NNS'), ('on', 'IN'), ('the', 'DT'), ('striped', 'JJ'),
# ('mat', 'NN'), ('under', 'IN'), ('many', 'JJ'), ('flying', 'VBG'), ('geese', 'JJ')]

# As you may have noticed, the above pos tags are a little confusing.

# we use our own pos_tagger function to make things simpler to understand.
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
print(wordnet_tagged)
#>[('the', None), ('cat', 'n'), ('is', 'v'), ('sitting', 'v'), ('with', None),
# ('the', None), ('bats', 'n'), ('on', None), ('the', None), ('striped', 'a'),
# ('mat', 'n'), ('under', None), ('many', 'a'), ('flying', 'v'), ('geese', 'a')]

lemmatized_sentence = []
for word, tag in wordnet_tagged:
	if tag is None:
		# if there is no available tag, append the token as is
		lemmatized_sentence.append(word)
	else:		
		# else use the tag to lemmatize the token
		lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
lemmatized_sentence = " ".join(lemmatized_sentence)

print(lemmatized_sentence)
#> the cat can be sit with the bat on the striped mat under many fly geese

###############################################################################
#3. TextBlob

'''
TextBlob is a python library used for processing textual data. It provides a simple API to access
its methods and perform basic NLP tasks.
'''

from textblob import TextBlob, Word

my_word = 'cats'

# create a Word object
w = Word(my_word)

print(w.lemmatize())
#> cat

sentence = 'the bats saw the cats with stripes hanging upside down by their feet.'

s = TextBlob(sentence)
lemmatized_sentence = " ".join([w.lemmatize() for w in s.words])

print(lemmatized_sentence)
#> the bat saw the cat with stripe hanging upside down by their foot

###############################################################################
#4. TextBlob (with POS tag)

'''
Same as in Wordnet approach without using appropriate POS tags, we observe the same
limitations in this approach as well. So, we use one of the more powerful aspects of the
TextBlob module the ‘Part of Speech’ tagging to overcome this problem.
'''

from textblob import TextBlob

# Define function to lemmatize each word with its POS tag

# POS_TAGGER_FUNCTION : TYPE 2
def pos_tagger(sentence):
	sent = TextBlob(sentence)
	tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
	words_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]	
	lemma_list = [wd.lemmatize(tag) for wd, tag in words_tags]
	return lemma_list

# Lemmatize
sentence = "the bats saw the cats with stripes hanging upside down by their feet"
lemma_list = pos_tagger(sentence)
lemmatized_sentence = " ".join(lemma_list)
print(lemmatized_sentence)
#> the bat saw the cat with stripe hang upside down by their foot

###############################################################################
#5. spaCy

import spacy
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(u'the bats saw the cats with best stripes hanging upside down by their feet')

# Create list of tokens from given string
tokens = []
for token in doc:
	tokens.append(token)

print(tokens)
#> [the, bats, saw, the, cats, with, best, stripes, hanging, upside, down, by, their, feet]

lemmatized_sentence = " ".join([token.lemma_ for token in doc])

print(lemmatized_sentence)
#> the bat see the cat with good stripe hang upside down by -PRON- foot

'''
In the above code, we observed that this approach was more powerful than our previous approaches as : 
Even Pro-nouns were detected. ( identified by -PRON-)
Even best was changed to good. 
'''

###############################################################################
#7. Pattern
'''
Pattern is a Python package commonly used for web mining, natural language processing,
machine learning and network analysis. It has many useful NLP capabilities.
It also contains a special feature which we will be discussing below.
'''

# PATTERN LEMMATIZER
import pattern
from pattern.en import lemma, lexeme
from pattern.en import parse

sentence = "the bats saw the cats with best stripes hanging upside down by their feet"

lemmatized_sentence = " ".join([lemma(word) for word in sentence.split()])

print(lemmatized_sentence)
#> the bat see the cat with best stripe hang upside down by their feet

# Special Feature : to get all possible lemmas for each word in the sentence
all_lemmas_for_each_word = [lexeme(wd) for wd in sentence.split()]
print(all_lemmas_for_each_word)

#> [['the', 'thes', 'thing', 'thed'],
# ['bat', 'bats', 'batting', 'batted'],
# ['see', 'sees', 'seeing', 'saw', 'seen'],
# ['the', 'thes', 'thing', 'thed'],
# ['cat', 'cats', 'catting', 'catted'],
# ['with', 'withs', 'withing', 'withed'],
# ['best', 'bests', 'besting', 'bested'],
# ['stripe', 'stripes', 'striping', 'striped'],
# ['hang', 'hangs', 'hanging', 'hung'],
# ['upside', 'upsides', 'upsiding', 'upsided'],
# ['down', 'downs', 'downing', 'downed'],
# ['by', 'bies', 'bying', 'bied'],
# ['their', 'theirs', 'theiring', 'theired'],
# ['feet', 'feets', 'feeting', 'feeted']]

###############################################################################
#8. Gensim
'''
Gensim is designed to handle large text collections using data streaming. 
Its lemmatization facilities are based on the pattern package we installed above. 

gensim.utils.lemmatize() function can be used for performing Lemmatization. This method comes under the utils module in python.
We can use this lemmatizer from pattern to extract UTF8-encoded tokens in their base form=lemma.
Only considers nouns, verbs, adjectives and adverbs by default (all other lemmas are discarded).
For example
Word          --->  Lemmatized Word 
are/is/being  --->  be
saw           --->  see
'''

from gensim.utils import lemmatize

sentence = "the bats saw the cats with best stripes hanging upside down by their feet"

lemmatized_sentence = [word.decode('utf-8').split('.')[0] for word in lemmatize(sentence)]

print(lemmatized_sentence)
#> ['bat / NN', 'see / VB', 'cat / NN', 'best / JJ',
# 'stripe / NN', 'hang / VB', 'upside / RB', 'foot / NN']

'''
In the above code as you may have already noticed, the gensim lemmatizer ignore the words
like ‘the’, ‘with’, ‘by’ as they did not fall into the 4 lemma categories mentioned above. (noun/verb/adjective/adverb)
'''

###############################################################################
###############################################################################
































