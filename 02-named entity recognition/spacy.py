# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:32:02 2021

@author: acer
"""

###############################################################################
###############################################################################
#spaCy

##pip install spacy
##python -m spacy download en

article = '''
Asian shares skidded on Tuesday after a rout in tech stocks put Wall Street to the sword, while a 
sharp drop in oil prices and political risks in Europe pushed the dollar to 16-month highs as investors dumped 
riskier assets. MSCI’s broadest index of Asia-Pacific shares outside Japan dropped 1.7 percent to a 1-1/2 
week trough, with Australian shares sinking 1.6 percent. Japan’s Nikkei dived 3.1 percent led by losses in 
electric machinery makers and suppliers of Apple’s iphone parts. Sterling fell to $1.286 after three straight 
sessions of losses took it to the lowest since Nov.1 as there were still considerable unresolved issues with the
European Union over Brexit, British Prime Minister Theresa May said on Monday.'''

import spacy

nlp = spacy.load("en_core_web_sm")
document = nlp(article)

print('Original Sentence: %s' % (article))

for element in document.ents:
    print(element.text + ' - ' + element.label_ + ' - ' + str(spacy.explain(element.label_)))
    
    
    
from spacy import displacy
sp = spacy.load('en_core_web_sm')

sen = sp(u'Manchester United is looking to sign Harry Kane for $90 million. David demand 100 Million Dollars')
displacy.render(sen, style='ent', jupyter=True)    
displacy.serve(sen, style='ent')

###############################################################################
###############################################################################


