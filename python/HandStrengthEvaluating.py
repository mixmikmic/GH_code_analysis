from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate, Card, Deck

import numpy as np
import pickle


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

hole_card = gen_cards(['H4', 'D7'])
community_card = gen_cards(['D3', 'C5', 'C6'])

get_ipython().run_cell_magic('time', '', 'estimate_hole_card_win_rate(nb_simulation=200, nb_player=9, hole_card=hole_card, community_card=None)')

get_ipython().run_cell_magic('time', '', 'means = []\nstds = []\n\nx = list(range(10, 1110, 100))\n\nfor i in x:\n    res = [estimate_hole_card_win_rate(nb_simulation=i, nb_player=9, hole_card=hole_card,\n                                       community_card=None) for j in range(10)]\n    means.append(np.mean(res))\n    stds.append(np.std(res))')

plt.errorbar(x, means, stds)

plt.errorbar(x, means, stds)

get_ipython().run_cell_magic('time', '', 'means = []\nstds = []\n\nx = list(range(2, 10))\n\nfor i in x:\n    res = [estimate_hole_card_win_rate(nb_simulation=200, nb_player=i, hole_card=hole_card,\n                                       community_card=community_card) for j in range(10)]\n    means.append(np.mean(res))\n    stds.append(np.std(res))')

plt.errorbar(x, means, stds)

suits = list(Card.SUIT_MAP.values())
ranks = list(Card.RANK_MAP.values())

get_ipython().run_cell_magic('time', '', 'scores = {}\n\nfor s1 in suits:\n    for r1 in ranks:\n        for s2 in suits:\n            for r2 in ranks:\n                card1 = s1 + r1\n                card2 = s2 + r2\n                if card1 == card2:\n                    continue\n                    \n                hole1 = (card1, card2)\n                hole2 = (card2, card1)\n                estimation = estimate_hole_card_win_rate(nb_simulation=10_000, nb_player=9,\n                                                         hole_card=gen_cards(hole1))\n                scores[hole1] = estimation\n                scores[hole2] = estimation')

with open('../cache/hole_card_estimation.pkl', 'wb') as f:
    pickle.dump(scores, f)

with open('../cache/hole_card_estimation.pkl', 'rb') as f:
    s = pickle.load(f)

len(s)

list(s.items())[:5]

plt.hist(list(s.values()), bins=30);

sorted(s.items(), key=lambda x: -x[1])[:5]

sorted(s.items(), key=lambda x: x[1])[:5]



