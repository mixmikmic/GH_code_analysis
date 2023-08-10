import glob
import pandas as pd
import numpy as np
from textblob import TextBlob
import pickle

def get_state(utter):
    """ Question, Sentiment, Subjectivity, Length"""
    utter = utter.split(":")[-1].strip("\n")
    state = []
    state.append(str(int(is_question(utter))))
    state.append(str(int(is_positive(utter))))
    state.append(str(int(is_long(utter))))
    state.append(str(int(is_subjective(utter))))
    return state
    
def is_question(utter):
    return utter[-1] == "?"

def is_positive(utter):
    senti = TextBlob(utter).sentiment.polarity
    if senti > 0.2:
        return 1
    elif senti < -0.2:
        return -1
    else:
        return 0

def is_subjective(utter):
    sub = TextBlob(utter).sentiment.subjectivity
    return sub > 0.5
    
def is_long(utter):
    return len(utter) > 20

def set_reward(df, lastline):
    ratings = lastline.strip("\n").strip(",").split(",")
    for rating in ratings:
        category, score = rating.split("=")
        df.set_value(df.index[-1], category, int(score))
    return df

filenames = glob.glob("../data/300_convo/*")
parsed_dict = {}
for filename in filenames:
    lines = open(filename, "r", errors="replace").readlines()
    df_sa = pd.DataFrame(columns=["Start", "Question", "Sentiment", "Length", "Subjective", "Previous", 
                                  "Action", "overall", "start", "interupt", "engaing", "return"])
    bot_lines = [i for i in lines[1:] if "Bot_" in i]
    user_lines = [i for i in lines if "_None_" in i]

    # initial state/action/reward
    action = lines[0].split("_")[1]
    df_sa.loc[0] = ["1","0","0","0","0","None",action,0,0,0,0,0] 
    previous_action = action
    
    # intermediate state/action/reward
    for i, (user, bot) in enumerate(zip(user_lines[:-1], bot_lines)):
        action = bot.split("_")[1]
        df_sa.loc[i+1] = ["0"] + get_state(user) + [previous_action, action] + [0]*5
        previous_action = action
        
    # terminal state/reward
    df_sa.loc[i+2] = ["T"]*6 + ["None"] + [0]*5
    df_sa = set_reward(df_sa.copy(), lines[-1])
    
    parsed_dict[filename] = df_sa

pickle.dump(parsed_dict, open("../data/parsed_data.pkl", "wb"))

print(open(filename, "r").read())

df_sa

parsed_dict["../data/300_convo/wei_2017-05-05_10.txt"]

def find_example(states):
    """ state is a list with four values:
    question (1, 0), sentiment (1, 0, -1), length (1, 0), previous(one of six actions), action
    """
    for filename, df in parsed_dict.items():
        if df.iloc[-1]["overall"] <=2:
            continue
        df_query = df[(df.Question == states[0])&
                      (df.Sentiment == states[1])&
                      (df.Length == states[2])&
                      (df.Previous == states[3])&
                      (df.Action==states[4])]
        if len(df_query) != 0:
            print(filename)
            print(df)

find_example(["0", "-1", "0", "question", "elaborate"])



