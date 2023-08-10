import numpy as np
import pandas as pd
import json
import re

# import all magic cards ever printed, from web 14mb

# allsets = pd.read_json('http://mtgjson.com/json/AllSetsArray.json')
# allsets.info()

# import all magic cards from file 

allsets = pd.read_json('data/AllSetsArray.json')
allsets.tail(2)

# drop columns that are full of nan and/or junk

allsets = allsets.drop(['booster', 'gathererCode', 'magicCardsInfoCode', 
                        'magicCardsInfoCode', 'code',
                        'oldCode', 'onlineOnly', 'magicRaritiesCodes'
                        ], axis=1)

# zoom in on cards from latest set, 'Battle for Zendikar'

zen = pd.read_json(json.dumps(allsets.cards[185]))

# add set name and date
zen['set'] = allsets.name[185]
zen['releaseDate'] = allsets.releaseDate[185]

zen.head(2)

# drop junk 
zen.drop(['id', 'layout', 'multiverseid', 'imageName', 'subtypes', 
          'supertypes', 'variations', 'loyalty', 'number'], 
         axis=1, inplace=True)

zen.head(2)

# grab the text from card 1

zen['text'][1]

# fix errors

zen['text'].replace("\n" , " ", inplace=True, regex=True)
zen['text'].replace("\'" , "", inplace=True, regex=True)

zen['text'][1]

# slice sets for all of modern (arbritrary rules change date in 2003)

modern_sets = allsets[ (allsets['releaseDate'] >= '2003-07-28' ) ]
modern_sets = modern_sets.loc[modern_sets['type'].isin(['core', 'expansion'])]
modern_sets = modern_sets.loc[modern_sets['border'].isin(['white', 'black'])]

modern_sets.reset_index(inplace=True)
modern_sets

# groups cards from sets into one data frame 

cards = None

for s in xrange(len(modern_sets)):
    print "Reading in set:", modern_sets.name[s]
    target = pd.read_json(json.dumps(modern_sets.cards[s]))
    
    # slice to just the good stuff 
    target = target[['artist', 'cmc', 'colors', 'flavor', 'manaCost', 'name', 
                    'power', 'rarity', 'text', 'toughness', 'type', 'types']]
    
    # add set name and date
    target['set'] = modern_sets.name[s]
    target['releaseDate'] = modern_sets.releaseDate[s]    
    
    # add to cards df 
    cards = pd.concat([cards, target])

# clean up errors
cards['text'].replace("\n" , " ", inplace=True, regex=True)
cards['text'].replace("\'" , "", inplace=True, regex=True)
cards['flavor'].replace("\n" , " ", inplace=True, regex=True)

print 
print cards.info(verbose=False)

# filter out lands and tokens 

cards = cards.loc[cards['rarity'].isin(['Common', 'Uncommon', 'Rare', 
                                        'Mythic Rare'])]
cards.info(verbose=False)

# only keep cards with a 'color' attribute
cards_no_nulls = cards[cards['colors'].notnull()]

# only keep cards with text
cards_no_nulls = cards_no_nulls[cards_no_nulls['text'].notnull()]

# only keep cards with a mana cost
cards_no_nulls = cards_no_nulls[cards_no_nulls['colors'].map(len) == 1]
cards_no_nulls.info(verbose=False)

# reset index and drop old index vals 

cards_no_nulls.reset_index(inplace=True)
cards_no_nulls.pop('index')
# cards_no_nulls.pop('level_0')
cards_no_nulls

# datamunge to get "color" out of a list format

def nolist(x):
    return x[0]

cards_no_nulls['colors'] = cards_no_nulls['colors'].apply(nolist)

# remove resource symbols from one card 

a = cards_no_nulls.text[3]
re.sub("{[A-Z]}" , "{1}", a)

# remove resource symbols from all cards 

def tap(x):
    return re.sub("{T}" , "Tap ", x)

def nomana(x):
    return re.sub("{[A-Z]}" , "{1}", x)

cards_no_nulls.text = cards_no_nulls.text.apply(tap)
cards_no_nulls.text = cards_no_nulls.text.apply(nomana)

cards_no_nulls.text

# store to disk

# cards_no_nulls.reset_index(inplace=True)
# cards_no_nulls.pop('index')

cards_no_nulls.to_pickle('data/cards_modern.pkl')

# remove name from one card 

re.sub(cards_no_nulls.name[3] , "This", cards_no_nulls.text[3] )

# remove card names from all cards  

import time

t0 = time.time()

for i in xrange(len(cards_no_nulls)):
    cards_no_nulls['text'][i] = re.sub(cards_no_nulls['name'][i] , "This", cards_no_nulls['text'][i]) 

t1 = time.time()

print round((t1-t0)/60, 2), "minutes"
cards_no_nulls.text

# store to disk

# cards_no_nulls.reset_index(inplace=True)
# cards_no_nulls.pop('index')

cards_no_nulls.to_pickle('data/cards_modern_no_name.pkl')

# remove helper text from one card 

a = cards_no_nulls.text[9]
re.sub('\([^)]*\)' , "", a)

# remove helper text 

def hardmode(x):
    return re.sub('\([^)]*\)' , "", x)

cards_no_nulls.text = cards_no_nulls.text.apply(hardmode)

# store to disk

cards_no_nulls.to_pickle('data/5color_modern_no_name_hardmode.pkl')

# groups all modern cards of all colors from sets into one data frame 

cards = None

for s in xrange(len(modern_sets)):
    print "Reading in set:", modern_sets.name[s]
    target = pd.read_json(json.dumps(modern_sets.cards[s]))
    
    # slice to just the good stuff 
    target = target[['artist', 'cmc', 'colors', 'flavor', 'manaCost', 'name', 
                    'power', 'rarity', 'text', 'toughness', 'type', 'types']]
    
    # add set name and date
    target['set'] = modern_sets.name[s]
    target['releaseDate'] = modern_sets.releaseDate[s]    
    
    # add to cards df 
    cards = pd.concat([cards, target])

# clean up errors
cards['text'].replace("\n" , " ", inplace=True, regex=True)
cards['text'].replace("\'" , "", inplace=True, regex=True)

print 
print cards.info(verbose=False)

# drop tokens 
cards_no_nulls = cards.loc[cards['rarity'].isin(['Common', 'Uncommon', 
                                                 'Rare', 'Mythic Rare'])]

# only keep cards with text
cards_no_nulls = cards_no_nulls[cards_no_nulls['text'].notnull()]

# save the "tap" symbol
cards_no_nulls.text = cards_no_nulls.text.apply(tap)

# flatten all resource symbols from all cards 
cards_no_nulls.text = cards_no_nulls.text.apply(nomana)

# remove helper text 
cards_no_nulls.text = cards_no_nulls.text.apply(hardmode)

# reset index
cards_no_nulls.reset_index(inplace=True)
cards_no_nulls.pop('index')

print "Done!"
cards_no_nulls.info()

# store to disk

cards_no_nulls.to_pickle('data/all_cards_modern_no_name_hardmode.pkl')



