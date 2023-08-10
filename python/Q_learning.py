import pandas as pd
from pprint import pprint as pp
import itertools
import pickle
import functools

get_ipython().magic('autosave 40')

dialogs = pickle.load(open("../data/parsed_data.pkl", "rb"))

class RL_BEGIN(object):
    def __init__(self, states=[], reward=[], gamma=0.9, alpha=0.5):
        self.states = states
        self.actions = ["question", "opinion", "elaborate", "affirmative", "negative", "neutral"]
        self.reward = reward
        self.alpha = alpha
        self.gamma = gamma
        self.q = self.initialize(states, self.actions)

    def initialize(self, states, actions):
        df = pd.DataFrame(columns=["State","Action","Value"],
                          data=list(itertools.product(["1"], actions,[0])))
        return df
    
    def update_q(self, dialogue, n_iters=1):
        """ update q table based on one dialogue, 
            returns: the intermediate q values during iteration, only works on first example
            in this RL_BEGIN class, dialogues only have a start and reward, middle is removed
        """
        dialogue_short = dialogue.drop(dialogue.index[1:-1])
        dialogue_short.reset_index(drop=True, inplace=True)
        q_mid = self.q.copy()
        q_mid.rename(columns={"Value": "iter0"}, inplace=True)
        for i in range(n_iters):
            q_mid["iter{0}".format(i+1)] = q_mid["iter{0}".format(i)].copy()
            for turn in dialogue_short.index:
                # get values of current state
                action = dialogue_short.loc[turn]["Action"]
                states = "".join(list(dialogue_short.loc[turn][self.states]))
                if "T" in states:
                    break
                q = float(self.q[(self.q.Action==action)&(self.q.State==states)]["Value"])
                reward = np.mean(list(dialogue_short.loc[turn][self.reward]))
                
                # get values of next state
                next_turn = dialogue_short.loc[turn+1]
                next_states = "".join(list(next_turn[self.states]))
                next_max_q = self.find_max_q(next_states, i)
                if "T" in next_states:
                    reward = np.mean(list(next_turn[self.reward]))
                    
                # update q based on current reward, current q and next state q
                q = q + self.alpha * (reward + self.gamma * next_max_q - q)
                self.q.loc[(self.q.Action==action)&(self.q.State==states), "Value"] = q
                q_mid.loc[(q_mid.Action==action)&(q_mid.State==states), "iter{0}".format(i+1)] = q
        return q_mid
    
    
    def find_max_q(self, next_states, iteration):
        if "T" in next_states:
            return 0
        else:
            q = self.q[(self.q.State==next_states)]
            return max(list(q["Value"]))

starter_RL = RL_BEGIN(states=["Start"], reward=["start"])
q_mid = starter_RL.update_q(example, n_iters=10)

example[["Start", "Action", "start"]]

starter_RL.q

q_mid

starter_RL = RL_BEGIN(states=["Start"], reward=["start"],
                actions=["question", "opinion", "elaborate", "affirmative", "negative", "neutral"])
for key, value in dialogs.items():
    print(".", end="")
    starter_RL.update_q(value, n_iters=10)

starter_RL.q

class RL_QA(object):
    def __init__(self, states=[], reward=[], gamma=0.9, alpha=0.5):
        self.states = states
        self.actions = ["question", "opinion", "elaborate", "affirmative", "negative", "neutral"]
        self.reward = reward
        self.alpha = alpha
        self.gamma = gamma
        self.q = self.initialize(states, self.actions)

    def initialize(self, states, actions):
        df = pd.DataFrame(columns=["State","Action","Value"],
                          data=list(itertools.product(["0", "1"], actions,[0])))
        return df
    
    def update_q(self, dialogue, n_iters=1):
        """ update q table based on one dialogue, 
            returns: the intermediate q values during iteration, only works on first example
        """
        q_mid = self.q.copy()
        q_mid.rename(columns={"Value": "iter0"}, inplace=True)
        for i in range(n_iters):
            q_mid["iter{0}".format(i+1)] = q_mid["iter{0}".format(i)].copy()
            for turn in dialogue.index:
                # get values of current state
                action = dialogue.loc[turn]["Action"]
                states = "".join(list(dialogue.loc[turn][self.states]))
                if "T" in states:
                    break
                q = float(self.q[(self.q.Action==action)&(self.q.State==states)]["Value"])
                reward = np.mean(list(dialogue.loc[turn][self.reward]))
                
                # get values of next state
                next_turn = dialogue.loc[turn+1]
                next_states = "".join(list(next_turn[self.states]))
                next_max_q = self.find_max_q(next_states, i)
                if "T" in next_states:
                    reward = np.mean(list(next_turn[self.reward]))
                    
                # update q based on current reward, current q and next state q
                q = q + self.alpha * (reward + self.gamma * next_max_q - q)
                self.q.loc[(self.q.Action==action)&(self.q.State==states), "Value"] = q
                q_mid.loc[(q_mid.Action==action)&(q_mid.State==states), "iter{0}".format(i+1)] = q
#             print(self.q)
        return q_mid
    
    
    def find_max_q(self, next_states, iteration):
        if "T" in next_states:
            return 0
        else:
            q = self.q[(self.q.State==next_states)]
            return max(list(q["Value"]))
        
def find_examples_w_user_question(dialogs):
    for filename, df in dialogs.items():
        if "1" in list(df["Question"]) and df.iloc[-1]["interupt"] > 3:
            return df

example = find_examples_w_user_question(dialogs)
question_RL = RL_QA(states=["Question"], reward=["interupt"])
q_mid = question_RL.update_q(example, n_iters=10)

example[["Question", "Action", "interupt"]]

question_RL.q

q_mid

question_RL = RL_QA(states=["Question"], reward=["interupt"])
for key, value in dialogs.items():
    print(".", end="")
    question_RL.update_q(value, n_iters=10)

question_RL.q

class RL(object):
    def __init__(self, reward=[], gamma=0.9, alpha=0.5):
        self.states = ["Question", "Sentiment", "Subjectivity", "Length"]
        self.actions = ["question", "opinion", "elaborate", "affirmative", "negative", "neutral"]
        self.reward = reward
        self.alpha = alpha
        self.gamma = gamma
        self.q = self.initialize(self.states, self.actions)

    def initialize(self, states, actions):
        df = pd.DataFrame(columns = self.states + ["Action","Value"],
                          data=list(itertools.product([0,1],[0,1],[0,1],[0,1],actions,[0])))
        return df
    
    def update_q(self, dialogue, n_iters=1):
        """ update q table based on one dialogue, 
            returns: the intermediate q values during iteration, only works on first example
        """
        q_mid = self.q.copy()
        q_mid.rename(columns={"Value": "iter0"}, inplace=True)
        for i in range(n_iters):
            q_mid["iter{0}".format(i+1)] = q_mid["iter{0}".format(i)].copy()
            for turn in dialogue.index:
                # get values of current state
                action = dialogue.loc[turn]["Action"]
                states = dialogue.loc[turn][self.states]
                if "T" in list(states):
                    break
                c1 = self.q.Action==action
                c2 = self.q.Question==int(states.Question)
                c3 = self.q.Sentiment==int(states.Sentiment)
                c4 = self.q.Subjectivity==int(states.Subjectivity)
                c5 = self.q.Length==int(states.Length)
                                            
                q = float(self.q[conjunction(c1,c2,c3,c4,c5)]["Value"])
                reward = np.mean(list(dialogue.loc[turn][self.reward]))
                # get values of next state
                next_turn = dialogue.loc[turn+1]
                next_states = "".join(list(next_turn[self.states]))
                next_max_q = self.find_max_q(next_states, i, query=conjunction(c2,c3,c4,c5))
                if "T" in next_states:
                    reward = np.mean(list(next_turn[self.reward]))
                    
                # update q based on current reward, current q and next state q
                q = q + self.alpha * (reward + self.gamma * next_max_q - q)
                self.q.loc[conjunction(c1,c2,c3), "Value"] = q
                
                m1 = q_mid.Action==action
                m2 = q_mid.Question==int(states.Question)
                m3 = q_mid.Sentiment==int(states.Sentiment)
                m4 = q_mid.Subjectivity==int(states.Subjectivity)
                m5 = q_mid.Length==int(states.Length)
                q_mid.loc[conjunction(m1,m2,m3,m4,m5), "iter{0}".format(i+1)] = q
        return q_mid
    
    
    def find_max_q(self, next_states, iteration, query=None):
        if "T" in next_states:
            return 0
        else:
            q = self.q[query]
            return max(list(q["Value"]))

                                            
def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)


def get_example(dialogs):
    for filename, df in dialogs.items():
        if df.iloc[-1]["overall"] > 3:
            return df

all_RL = RL(reward=["overall"])

example = get_example(dialogs)
q_mid = all_RL.update_q(example, n_iters=10)

example[all_RL.states + ["Action", "overall"]]

all_RL.q[(all_RL.q.Question==0) &(all_RL.q.Length==1) & (all_RL.q.Sentiment==1)]

all_RL.q.sort_values(by="Value", ascending=False)

all_RL = RL(reward=["overall"])
for key, value in dialogs.items():
    print(".", end="")
    all_RL.update_q(value, n_iters=10)

all_RL.q.sort_values(by="Value", ascending=False)



