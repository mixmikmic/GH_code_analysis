import random
import itertools
from pprint import pprint as pp

user_utters = ["Hey", "Hmmm", "Haha", "Yo", "red", "25", "Just chilling", # answers
               "Hi?", "Really?", "Why?", "What's up?", "Are you smart?"] # questions
bot_utters = ["Right", "Yes", "No", "I don't know", "Maybe", "Hi", "I am good", "Never mind", # answers
              "How are you?", "What?", "What's that?", "Why not?", "Excuse me?", "Can you elaborate?"] # questions

# some helper functions
def is_question(utter):
    return utter[-1] == "?"

def get_question(utters):
    answers = [i for i in utters if i[-1] != "?"]
    questions = [i for i in utters if i[-1] == "?"]
    return random.choice(questions)

def get_answer(utters):
    answers = [i for i in utters if i[-1] != "?"]
    questions = [i for i in utters if i[-1] == "?"]
    return random.choice(answers)

def user_respond(bot_utter, mode="logically"):
    if mode=="logically":
        if is_question(bot_utter):
            return get_answer(user_utters)
        else:
            return get_question(user_utters)
    elif mode=="randomly":
        return random.choice(user_utters)

for turn in range(10):
    bot_utter = random.choice(bot_utters)
    print("Bot: ", bot_utter)
    print("User: ", user_respond(bot_utter))

# define transition function
def transition(s, a, user_behavior="logically"):
    """
    given a state s and an action a, return reward and a new state s_new:
    s: True or False (is question)
    a: "Q" or "A"
    s_new: True or False
    reward: 0 or 1
    """
    # calculate reward
    if (s and a == "Q"):
        reward = 0
    elif (s and a == "A"):
        reward = 1
    elif (not s and a == "Q"):
        reward = 1
    elif (not s and a == "A"):
        reward = 0
    else:
        raise
    
    # get new state
    if a == "Q":
        bot_utter = get_question(bot_utters)
    elif a == "A":
        bot_utter = get_answer(bot_utters)
    user_utter = user_respond(bot_utter, mode=user_behavior)
    s_new = is_question(user_utter)
        
    return reward, s_new


def carry_out_best_policy(user_say, best_actions):
    print("User: ", user_say)
    for action in best_actions:
        if action == "Q":
            bot_say = get_question(bot_utters)
        elif action == "A":
            bot_say = get_answer(bot_utters)
        else:
            raise
        print("Bot: ", bot_say)
        print("User: ", user_respond(bot_say))

# reinforcement learning function
def RL(user_say, n_turn = 5, user_behavior="logically"):
    actions = ["Q", "A"]
    rewards = []
    all_action_combinations = itertools.product(["Q", "A"], repeat=n_turn)
    for each_action_sequence in all_action_combinations:
        state = is_question(user_say)
        reward = 0
        for each_action in each_action_sequence:
            r, state = transition(state, each_action, user_behavior=user_behavior)
            reward += r
        rewards.append(("".join(each_action_sequence), reward))
    
    print("all possible action rewards: ")
    pp(rewards)
    best_policy = sorted(rewards, key=lambda x:x[1], reverse=True)[0][0]
    print("best action sequence: ", best_policy)
    carry_out_best_policy(user_say, best_policy)

RL("How are you?", user_behavior="logically")

RL("How are you?", user_behavior="randomly")

RL("Not a question", user_behavior="logically")

RL("Not a question", user_behavior="randomly")



