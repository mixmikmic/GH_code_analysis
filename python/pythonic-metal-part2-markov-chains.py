import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn

get_ipython().magic('matplotlib inline')

from utils import tokenizer, colouring
from wordcloud import WordCloud
import nltk
from nltk import FreqDist

lyrics = pd.read_csv("data/lyrics.csv")

# load some bands to focus on
with open("data/artists.txt", "r") as f:
    bands = f.read().strip().replace('"', "").split("\n")
    f.close()
    
lyrics = lyrics[lyrics.band_name.isin(bands)].copy()    

START_OF_SEQ = "~"
END_OF_SEQ = "[END]"

import random
import json

class MarkovChain:
    """
    Simple Markov Chain Class
    """

    def __init__(self, order=1, pad=True, records=None):
        """
        Initialise Markov chain
        :param order: int - number of tokens to consider a state
        :param pad: bool - whether to pad training strings with start/end tokens
        """
        self.order = order
        self.pad = pad
        self.records = {} if records is None else records

    def add_tokens(self, tokens):
        """
        Adds a list of tokens to the markov chain

        :param tokens: list of tokens
        :return: None
        """
        if self.pad:
            tokens = [START_OF_SEQ] * self.order + tokens + [END_OF_SEQ]

        for i in range(len(tokens) - self.order):
            current_state = tuple(tokens[i:i + self.order])
            next_state = tokens[i + self.order]
            self.add_state(current_state, next_state)

    def add_state(self, current_state, next_state):
        """
        Updates the weight of the transition from current_state to next_state
        with a single observation.

        :param current_state: tuple - current state
        :param next_state: token - the next observed token
        :return: None
        """
        if current_state not in self.records.keys():
            self.records[current_state] = dict()

        if next_state not in self.records[current_state].keys():
            self.records[current_state][next_state] = 0

        self.records[current_state][next_state] += 1

    def generate_sequence(self, n=100, initial_state=None):
        """
        Generates a sequence of tokens from the markov chain, starting from
        initial_state. If initial state is empty, and pad is false it chooses an
        initial state at random. If pad is true,

        :param n: int - The number of tokens to generate
        :param initial_state: starting state of the generator
        :return: list of generated tokens
        """

        if initial_state is None:
            if self.pad:
                sequence = [START_OF_SEQ] * self.order
            else:
                sequence = list(random.choice(self.records.keys()))
        else:
            sequence = initial_state[:]

        for i in range(n):
            current_state = tuple(sequence[-self.order:])
            next_token = self.sample(current_state)
            sequence.append(next_token)

            if next_token == END_OF_SEQ:
                return sequence

        return sequence

    def sample(self, current_state):
        """
        Generates a random next token, given current_state
        :param current_state: tuple - current_state
        :return: token
        """

        possible_next = self.records[current_state]
        n = sum(possible_next.values())

        m = random.randint(0, n)
        count = 0
        for k, v in possible_next.items():
            count += v
            if m <= count:
                return k

    def save(self, filename):
        """
        Saves Markov chain to filename

        :param filename: string - where to save chain
        :return: None
        """
        with open(filename, "w") as f:
            m = {
                "order": self.order,
                "pad": self.pad,
                "records": {str(k): v for k, v in self.records.items()}
            }
            json.dump(m, f)

    @staticmethod
    def load(filename):
        """
        Loads Markov chain from json file
        
        DUE TO USE OF EVAL
        DO NOT RUN THIS ON UNTRUSTED FILES
        
        :param filename: 
        :return: MarkovChain
        """
        with open(filename, "r") as f:
            raw = json.load(f)

        mc = MarkovChain(
            raw["order"], 
            raw["pad"], 
            {eval(k):v for k,v in raw["records"].items()}
        )
        
        return mc

notb = lyrics[lyrics.song_name == " The Number Of The Beast"].lyrics.values[0]
print(notb)

mc_1_word_notb = MarkovChain(1, pad=False)
mc_1_word_notb.add_tokens(tokenizer.tokenize_strip_non_words(notb)[71:127])

import graphviz
gv = graphviz.Digraph(engine="dot", format="png")

for n, e in mc_1_word_notb.records.items():
    gv.node(n[0])
    for k,v in e.items():
        gv.edge(n[0], k)
        
gv.render("resources/666")

gv

mc_4_words = MarkovChain(4, pad=True)

for s in lyrics.lyrics.values:
    mc_4_words.add_tokens(tokenizer.tokenize_words(s))

full_text = tokenizer.normalise("".join(lyrics.lyrics.values))

def is_in_lyrics(s):
    return s in full_text

def colour_based_on_existance(s):
    col = colouring.ColourIter()

    n = len(s)
    start = 0
    end = 1
    completed = []

    while end < n:
        if not is_in_lyrics(s[start:end]):
            completed.append(colouring.colour_text_background_html(s[start:end], col()))
            start = end
        else:
            end += 1
        
    completed.append(colouring.colour_text_background_html(s[start:], col()))
        
    return completed

from IPython.core.display import display, HTML

samp = tokenizer.tokenized_pretty_print(mc_4_words.generate_sequence(n=200))

htmled = "<p>{}</p>".format("".join(colour_based_on_existance(samp)).replace("\n", "<br>"))

display(HTML(htmled))

print(htmled)



