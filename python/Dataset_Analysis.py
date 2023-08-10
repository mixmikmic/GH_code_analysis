get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import json

with open("train_full.json") as f:
    dataset = json.load(f)

# dataset[50]["thread"] # print some data

long_dialogs = 0
dialog_lens = []

for d in dataset:
    Alice = 0
    Bob = 0
    for u in d["thread"]:
        if u["userId"] == "Alice":
            Alice += 1
        elif u["userId"] == "Bob":
            Bob += 1
        else: 
            raise ValueError
    
    dialog_lens.append(Alice + Bob)
    if Alice > 2 and Bob > 2:
        long_dialogs += 1

long_dialogs

plot = plt.hist(dialog_lens, bins=10, range=(0,50))

# dataset[200] # print some data

human_lens = []
bot_lens = []
for d in dataset:
    for user in d["users"]:
        if user["id"] == "Alice":
            Alice = True if user["userType"] == "Human" else False
        elif user["id"] == "Bob":
            Bob = True if user["userType"] == "Human" else False
    for u in d["thread"]:
        if u["userId"] == "Alice":
            if Alice:
                human_lens.append(len(u["text"]))
            else:
                bot_lens.append(len(u["text"]))
        elif u["userId"] == "Bob":
            if Bob:
                human_lens.append(len(u["text"]))
            else:
                bot_lens.append(len(u["text"]))

plot = plt.hist((human_lens,bot_lens), bins=10, normed = 1, range=(0,200), label = ('human','bot'))
l = plt.legend()

human_eval = []
bot_eval = []
for d in dataset:
    for user in d["users"]:
        if user["id"] == "Alice":
            Alice = True if user["userType"] == "Human" else False
        elif user["id"] == "Bob":
            Bob = True if user["userType"] == "Human" else False
    for u in d["thread"]:
        if u["userId"] == "Alice":
            if Alice:
                human_eval.append((u["evaluation"]-1))
            else:
                bot_eval.append((u["evaluation"]-1))
        elif u["userId"] == "Bob":
            if Bob:
                human_eval.append((u["evaluation"]-1))
            else:
                bot_eval.append((u["evaluation"]-1))

p = plt.hist((human_eval,bot_eval), bins=3,  normed = (1,1), label = ('human','bot'))
l = plt.legend()

human_quality = []
bot_quality = []
for d in dataset:
    for user in d["users"]:
        if user["id"] == "Alice":
            Alice = True if user["userType"] == "Human" else False
        elif user["id"] == "Bob":
            Bob = True if user["userType"] == "Human" else False
    for u in d["evaluation"]:
        if u["quality"] > 0:
            if u["userId"] == "Alice":
                if Alice:
                    human_quality.append(u["quality"])
                else:
                    bot_quality.append(u["quality"])
            elif u["userId"] == "Bob":
                if Bob:
                    human_eval.append(u["quality"])
                else:
                    bot_eval.append(u["quality"])

# dataset[200]

p = plt.hist((human_quality,bot_quality),  bins=5, normed = (1,1), label = ('human','bot'))
l = plt.legend()

human_quality = []
bot_quality = []
for d in dataset:
    for user in d["users"]:
        if user["id"] == "Alice":
            Alice = True if user["userType"] == "Human" else False
        elif user["id"] == "Bob":
            Bob = True if user["userType"] == "Human" else False
    for u in d["evaluation"]:
        if u["engagement"] > 0:
            if u["userId"] == "Alice":
                if Alice:
                    human_quality.append(u["engagement"])
                else:
                    bot_quality.append(u["engagement"])
            elif u["userId"] == "Bob":
                if Bob:
                    human_eval.append(u["engagement"])
                else:
                    bot_eval.append(u["engagement"])

p = plt.hist((human_quality,bot_quality),  bins=5, normed = (1,1), label = ('human','bot'))
l = plt.legend()

human_quality = []
bot_quality = []
for d in dataset:
    for user in d["users"]:
        if user["id"] == "Alice":
            Alice = True if user["userType"] == "Human" else False
        elif user["id"] == "Bob":
            Bob = True if user["userType"] == "Human" else False
    for u in d["evaluation"]:
        if u["engagement"] > 0:
            if u["userId"] == "Alice":
                if Alice:
                    human_quality.append(u["breadth"])
                else:
                    bot_quality.append(u["breadth"])
            elif u["userId"] == "Bob":
                if Bob:
                    human_eval.append(u["breadth"])
                else:
                    bot_eval.append(u["breadth"])

p = plt.hist((human_quality,bot_quality),  bins=5, normed = (1,1), label = ('human','bot'))
l = plt.legend()

