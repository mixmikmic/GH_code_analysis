from wowp.actors import FuncActor
from wowp.schedulers import ThreadedScheduler
from wowp.components import draw_graph

get_ipython().run_line_magic('pylab', 'inline --no-import-all')

# How many branching will be there:
power = 8

# Basic building blocks
def add(a, b) -> ('a'):
    return a + b

leaves = []

def split(act, depth):
    global leaves
    if depth == 0:
        leaves.append(act)
    else:
        child1 = FuncActor(add)
        child2 = FuncActor(add)
        child1.outports.a.connect(act.inports.a)
        child2.outports.a.connect(act.inports.b)
        split(child1, depth-1)
        split(child2, depth-1)

# Create the actor
last = FuncActor(add)
split(last, power)

# Let's draw the graph of actors
graph = last.graph
plt.subplots(figsize=(12, 8))
draw_graph(graph)

# from wowp.tools.plotting import ipy_show (old style)
# ipy_show(last)

# We have 64 inports, let's sum number 1..64

# (in threads)
scheduler = ThreadedScheduler(max_threads=8)

for i, actor in enumerate(leaves):
    scheduler.put_value(actor.inports.a, i * 2 + 1)
    scheduler.put_value(actor.inports.b, i * 2 + 2)

scheduler.execute()
print("Result: ", last.outports.a.pop())

# Check the result (obtained in a somewhat ;-) better way)
sum(range(1, 2 * 2**power + 1))

