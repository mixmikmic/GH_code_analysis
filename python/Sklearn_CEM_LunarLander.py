import gym, gym.wrappers
gym.logger.level=0 #gym.youre("drunk").shut_up()
import numpy as np
from sklearn.neural_network import MLPClassifier

#Create environment
env = gym.make("LunarLander-v2")
n_actions = env.action_space.n


#Create agent
agent = MLPClassifier(hidden_layer_sizes=(256,512),
                      activation='tanh',
                      solver='adam',
                      warm_start=True,max_iter=1
                     )
#initialize agent by feeding it with some random bullshit
agent.fit([env.reset()]*n_actions,range(n_actions));

from itertools import count
def generate_session(t_max=10**3):
    """
    Just ask agent to predict action and see how env reacts - repeat until exhaustion.
    :param t_max: after this many steps the session is forcibly stopped. MAKE SURE IT'S ENOUGH!"""
    states,actions,total_reward = [],[],0
    
    s = env.reset()    
    for t in count():
        a = np.random.choice(n_actions,p=agent.predict_proba([s])[0])
        states.append(s)
        actions.append(a)
        
        s,r,done,_ = env.step(a)
        total_reward+=r
        
        if done or t>t_max:break
    return states,actions,total_reward

from joblib import Parallel,delayed
generate_sessions = lambda n,n_jobs=-1: Parallel(n_jobs)(n*[delayed(generate_session)()])

#training loop
#if you want faster stochastic iterations, try n_samples=100,percentile=50~70. Also maybe tune learning rate.
n_samples = 500   #takes 500 samples
percentile = 80   #fits to 20% best (100 samples) on each epoch
n_jobs = -1       #uses all cores


for i in range(150):
    #sample sessions
    sessions = generate_sessions(n_samples,n_jobs)
    batch_states,batch_actions,batch_rewards = map(np.array,zip(*sessions))
    
    #choose threshold on rewards
    threshold = np.percentile(batch_rewards,percentile)
    elite_states = np.concatenate(batch_states[batch_rewards>=threshold])
    elite_actions = np.concatenate(batch_actions[batch_rewards>=threshold])
    
    #fit our osom neural network >.<
    agent.fit(elite_states,elite_actions)

    #report progress
    print("epoch %i \tmean reward=%.2f\tthreshold=%.2f"%(i,batch_rewards.mean(),threshold))

#if you run on a headless server, run this
get_ipython().system('bash xvfb start')
get_ipython().magic('env DISPLAY=:1')

#finish recording
env = gym.wrappers.Monitor(env,directory="videos",force=True)
sessions = [generate_session() for _ in range(500)]
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



