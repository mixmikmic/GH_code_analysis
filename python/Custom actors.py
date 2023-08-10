from wowp import Actor

class StrActor(Actor):

    def __init__(self, *args, **kwargs):
        super(StrActor, self).__init__(*args, **kwargs)
        # specify input port
        self.inports.append('input')
        # and output ports
        self.outports.append('output')
        
    def get_run_args(self):
        # get input value(s) using .pop()
        args = (self.inports['input'].pop(), )
        kwargs = {}
        return args, kwargs

    @staticmethod
    def run(value):
        # return a dictionary with port names as keys
        res = {'output': str(value)}
        return res

actor = StrActor(name='str_actor')

# we can call the actor directly -- see what's output
value = 123
print(actor(input=value))

# and check that the output is as expected
assert actor(input=value)['output'] == str(value)

from wowp.actors import FuncActor

# use randint as input to out StrActor
import random
rand = FuncActor(random.randint)

actor.inports['input'] += rand.outports['out']

# get the workflow
wf = actor.get_workflow()

# and execute
wf(a=0, b=5)

