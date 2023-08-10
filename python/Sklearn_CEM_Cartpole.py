#if you run on a headless server, run this
get_ipython().system('bash xvfb start')
get_ipython().magic('env DISPLAY=:1')

import gym, gym.wrappers
gym.logger.level=0 #gym.youre("drunk").shut_up()
import numpy as np
from sklearn.neural_network import MLPClassifier

#Create environment
env = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env,directory="videos",force=True)

n_actions = env.action_space.n


#Create agent
agent = MLPClassifier(hidden_layer_sizes=(20,20),
                      activation='tanh',
                      solver='adam',
                      warm_start=True,max_iter=1
                     )
#initialize agent by feeding it with some random bullshit
agent.fit([env.reset()]*n_actions,range(n_actions));

def generate_session():
    """
    Just ask agent to predict action and see how env reacts - repeat until exhaustion.
    :param greedy: if True, picks most likely actions, else samples actions"""
    states,actions,total_reward = [],[],0
    
    s = env.reset()    
    while True:
        a = np.random.choice(n_actions,p=agent.predict_proba([s])[0])
        
        states.append(s)
        actions.append(a)
        
        s,r,done,_ = env.step(a)
        total_reward+=r
        if done:break
        
    return states,actions,total_reward
        

#training loop
n_samples = 100 #take 100 samples
percentile = 70 #fit on top 30% (30 best samples)

for i in range(50):
    #sample sessions
    sessions = [generate_session() for _ in range(n_samples)]
    batch_states,batch_actions,batch_rewards = map(np.array,zip(*sessions))
    
    #choose threshold on rewards
    threshold = np.percentile(batch_rewards,percentile)
    elite_states = np.concatenate(batch_states[batch_rewards>=threshold])
    elite_actions = np.concatenate(batch_actions[batch_rewards>=threshold])
    
    #fit our osom neural network >.<
    agent.fit(elite_states,elite_actions)

    #report progress
    print("epoch %i \tmean reward=%.2f\tthreshold=%.2f"%(i,batch_rewards.mean(),threshold))

#finish recording
env.close()
gym.upload("./videos/",api_key="<...>")

from IPython.display import HTML
import os

video_names = list(filter(lambda s:s.endswith(".mp4"),os.listdir("./videos/")))

HTML("""
<video width="640" height="480" controls>
  <source src="{}" type="video/mp4">
</video>
""".format("./videos/"+video_names[-1])) #this may or may not be _last_ video. Try other indices



