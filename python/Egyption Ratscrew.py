from collections import namedtuple
from itertools import chain

Card = namedtuple('Card', ['face', 'suite'])
royal = {'Jack': 1, 'Queen': 2, 'King': 3, 'Ace': 4}


def new_deck():
    cards = []
    for face in chain(range(2, 10 + 1), royal):
        for suite in ('Clubs', 'Diamonds', 'Hearts', 'Spades'):
            card = Card(face, suite)
            cards.append(card)
    return cards


assert len(new_deck()) == 52

from random import shuffle
from collections import deque


def deal(deck, num_players):
    """
    Shuffle a deck, and deal its cards equally amoung `num_players` hands.

    Args:
        deck: List[Card]
        num_players: int > 0

    Returns:
        List[collections.deque]
    """
    shuffle(deck)

    # Since we add the pile to the bottomfdasf  
    hands = [deque() for _ in range(num_players)]                     

    for hand in cycle(hands):
        if not deck:
            return hands
        new_card = deck.pop()
        hand.append(new_card)

    return hands

def slappable(pile):
    if len(pile) > 1 and pile[-1].face == pile[-2].face:
        return True
    if len(pile) > 2 and pile[-1].face == pile[-3].face:
        return True
    return False


assert not slappable([])

for card in new_deck():
    assert not slappable([card])

assert slappable([Card('Queen', 'Diamonds'), Card('Queen', 'Hearts')])
assert slappable([Card(2, 'Spades'), Card(2, 'Clubs')])

assert not slappable([Card('Queen', 'Diamonds'), Card('King', 'Hearts')])
assert not slappable([Card(2, 'Spades'), Card(3, 'Clubs')])

assert slappable([Card('Queen', 'Diamonds'), Card(2, 'Spades'), Card('Queen', 'Hearts')])
assert slappable([Card(2, 'Spades'), Card('Queen', 'Diamonds'), Card(2, 'Clubs')])

assert not slappable([Card('Queen', 'Diamonds'), Card(2, 'Spades'), Card('King', 'Hearts')])
assert not slappable([Card(2, 'Spades'), Card('Queen', 'Diamonds'), Card(3, 'Clubs')])

from itertools import cycle
import logging

logger = logging.getLogger(__name__)

# import random
# np.random.seed(42)
# random.seed(42)

def simulate_game(num_players, slap):
    deck = new_deck()
    hands = deal(deck, num_players)  # hands are stacks (cards face down)
    pile = []                        # pile is a stack (cards face up)
    chances = 0  # number of chances for current challenged player
    challenger = None  # index of player who played royal card

    for turn, hand_index in enumerate(cycle(range(num_players)), start=1):            
        hand = hands[hand_index]
        
        logger.info('{} {} {}'.format(
            [len(hand) for hand in hands],
            chances,
            [card.face for card in pile]
        ))

        if sum(len(hand) > 0 for hand in hands) == 1:
            return turn, np.argmax([len(h) for h in hands])
        if not hand:
            continue

        if challenger is None:
            new_card = hand.pop()
            pile.append(new_card)
            logger.info('player {} drew a {}'.format(hand_index, new_card.face))
            if new_card.face in royal:
                challenger = hand_index
                chances = royal[new_card.face]
        else:
            while chances > 0 and hand:
                new_card = hand.pop()
                logger.info('player {} drew a {}'.format(hand_index, new_card.face))
                pile.append(new_card)
                chances -= 1

                # A player can slap the pile even if a royal was placed.
                if slappable(pile):
                    slapper = slap()
                    logger.info('player {} slapped'.format(slapper))
                    hands[slapper].extendleft(reversed(pile))
                    pile.clear()
                    chances = 0
                    break

                # Break out of challenged if royal is placed.
                if new_card.face in royal:
                    challenger = hand_index
                    chances = royal[new_card.face]
                    break

            if chances == 0:
                hands[challenger].extendleft(reversed(pile))
                pile.clear()
                challenger = None

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
get_ipython().magic('matplotlib inline')


num_players = 2

# def slap():
#     return np.argmax(np.random.multinomial(1, [1/num_players]*num_players))

def slap():
    return np.argmax(np.random.multinomial(1, [0.2, 0.8]))
    

turns = [simulate_game(num_players, slap)[0] for _ in range(500)]
plt.hist(turns)
plt.show()

from scipy.stats import describe

num_simulations = 100
num_players = 2

turns = [simulate_game(num_players, .1) for _ in range(num_simulations)]
turns = [t for t in turns if t is not None]
describe(turns)

