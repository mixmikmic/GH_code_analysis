from numpy.random import random_sample
from numpy import dot
from numpy.linalg import norm, det

from wowp.actors import FuncActor

dims = 4, 4
A = random_sample(dims)
B = random_sample(dims)

dot_actor = FuncActor(dot, inports=('a', 'b'))
det_actor = FuncActor(det)

det_actor.inports['a'] += dot_actor.outports['out']
wf = det_actor.get_workflow()

wf_res = wf(a=A, b=B)
assert wf_res['out'][0] == det(dot(A, B))

from wowp.schedulers import IPyClusterScheduler
ipyscheduler = IPyClusterScheduler()

wf_res = wf(scheduler=ipyscheduler, a=A, b=B)
assert wf_res['out'][0] == det(dot(A, B))

