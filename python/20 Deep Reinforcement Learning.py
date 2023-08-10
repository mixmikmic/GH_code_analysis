get_ipython().magic('pylab inline')
import environment 
import replay 
import agent
from time import sleep
from IPython.display import display, clear_output
from pprint import pprint as pp

reload(environment)
reload(replay)
reload(agent)
        
env = environment.Environment('Breakout-v0')
replay = replay.ExperienceReplay(env)
agent = agent.Agent(env, replay)

action = env.random_action()
screen, reard, done, info = env.step(action)

pylab.imshow(screen, cmap='Greys_r')
print screen.shape

