from wowp.actors.matlab import MatlabMethod

ceil = MatlabMethod('ceil', inports='x')

print(ceil(3.1))

from wowp.actors import FuncActor

# create a simple +1 actor
def add_one(x) -> ('y'):
    return x+1

add_one_actor = FuncActor(add_one)

# connect actors
ceil.inports['x'] += add_one_actor.outports['y']
# get the workflow object
wf = ceil.get_workflow()

# run the workflow
wf(x=2.1)

