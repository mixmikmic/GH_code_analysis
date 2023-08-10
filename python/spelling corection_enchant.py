import pandas as pd
import json

import numpy as np
import matplotlib.pyplot as plt

import pylab
import nltk
import operator 
from collections import Counter
import regex as re

import re
import string



######## Enchant & spelling # need to have 32 bit version : https://stackoverflow.com/questions/33709391/using-multiple-python-engines-32bit-64bit-and-2-7-3-5
import enchant
import wx
from enchant.checker import SpellChecker
from enchant.checker.wxSpellCheckerDialog import wxSpellCheckerDialog
from enchant.checker.CmdLineChecker import CmdLineChecker

#### data
fr_filename='C:/Users/Robin/Dropbox/Travaux_&_Rapports_Stages/UbicomLab-GATech/MeToo/code/data/#balancetonporc/tweet_fr.json'
df_fr = pd.read_json(fr_filename, orient="columns")


en_filename = 'C:/Users/Robin/Dropbox/Travaux_&_Rapports_Stages/UbicomLab-GATech/MeToo/code/data/#metoo/tweet_en.json'
df_en = pd.read_json(en_filename, orient="columns")

pwl = enchant.request_pwl_dict("my_words.txt")
def spelling(text,langue):#langue=  "en_GB": British English ;"en_US": American English; "de_DE": German, "fr_FR": French
    d = enchant.DictWithPWL("fr","my_words.txt")
    chkr = enchant.checker.SpellChecker(d)
    chkr.set_text(text)
    print(text)
    for err in chkr:
        #print(err.word)
        sug = err.suggest()[0]
        err.replace(sug)
    
    text = chkr.get_text()#returns corrected text
    #print(text)
    return(c)

def correct_speling(df,lang): # Thus method is working, but this is not really efficient
    i=0
    e=0
    while i <len((df['text'])):
        try: 
            text=df['text'].iloc[i]
            spelling(text,lang)
        except :
            e+=1
        i+=1
    print('there is',e,'errors')

correct_speling(df_en,"en_US")
correct_speling(df_fr,"en_FR")

df_en.to_json('tweet_en_corrected.json')
df_fr.to_json('tweet_fr_corrected.json')

a = "Ceci est un text avec beuacuop d'ereurs et pas snychro"
df_p_btp_test=df_p_btp[:100]
correct_speling(df_p_btp_test)

print(df_test['text'])

