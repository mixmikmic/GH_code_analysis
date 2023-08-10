import collections

# construct a simple class to represent individual cards 
# object with just attributes with no custom methods
Card = collections.namedtuple('Card', ['rank','suit'])







# first card

# last card

from random import choice









suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high):
    print(card)



