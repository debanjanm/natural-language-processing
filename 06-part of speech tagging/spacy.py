# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:27:42 2021

@author: acer
"""

###############################################################################
###############################################################################

import spacy
sp = spacy.load('en_core_web_sm')

sen = sp(u"I like to play football. I hated it in my childhood though")


for word in sen:
    print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
    
'''
Visualizing POS tags in a graphical way is extremely easy. 
The displacy module from the spacy library is used for this purpose. 
'''
from spacy import displacy

sen = sp(u"I like to play football. I hated it in my childhood though")
displacy.render(sen, style='dep', jupyter=True, options={'distance': 85}) 
displacy.serve(sen, style='dep', options={'distance': 120})  

###############################################################################
############################################################################### 