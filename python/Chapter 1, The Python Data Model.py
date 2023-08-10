import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in list(range(2, 11)) + list('JQKA')]
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                      for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

deck = FrenchDeck()

len(deck)

deck[17]

from random import choice

choice([3,4,6,7,8])  # picks 1 from a list

choice(deck)

choice(deck)

deck[:12]

for count, card in enumerate(deck):
    print(card)
    if count > 6:
        break

for count, card in enumerate(reversed(deck)):
    print(card)
    if count > 6:
        break

Card('Q', 'hearts') in deck

Card('Q', 'dogs') in deck

# sorting

suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high):
    print(card)

from math import hypot

class Vector:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector(%r, %r)' % (self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

Vector(4,5) + Vector(1,2)

Vector(1,2) * 3

bool(Vector(4,5))

repr(Vector(1,2))

str(Vector(1,2))

