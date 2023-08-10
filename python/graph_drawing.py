get_ipython().run_line_magic('pylab', 'inline --no-import-all')

from wowp import *
from wowp.components import draw_graph

from wowp.actors import FuncActor

def func1(x) -> ('a'):
    return x * 2

def func2(x, y) -> ('a'):
    return x + y

# create function actors
in_actor1 = FuncActor(func1)
in_actor2 = FuncActor(func1)
out_actor = FuncActor(func2)

# connect actors
out_actor.inports['x'] += in_actor1.outports['a']
out_actor.inports['y'] += in_actor2.outports['a']

graph = out_actor.graph

plt.subplots(figsize=(12, 8))
draw_graph(graph)

