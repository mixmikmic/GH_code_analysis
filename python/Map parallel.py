from wowp.actors import FuncActor
from wowp.schedulers import LinearizedScheduler, IPyClusterScheduler
from wowp.actors.mapreduce import Map

# numpy will perform some calculations
import numpy as np

map_actor = Map(
    FuncActor,
    args=(np.sin, ),
    scheduler=LinearizedScheduler())

inp = np.linspace(0, np.pi, 10)

results = map_actor(inp=inp)
assert np.allclose(np.sin(inp), results['out'])

map_parallel = Map(
    FuncActor,
    args=(np.sin, ),
    scheduler=IPyClusterScheduler())

results_p = map_parallel(inp=inp)
assert np.allclose(np.sin(inp), results_p['out'])

from wowp.actors.mapreduce import PassWID

# this will be our current process
PassWID()(inp=1)

pid_act = Map(PassWID, scheduler=LinearizedScheduler())

# With LinearizedScheduler, this will still be the same process.
pid_act(inp=range(3))['out']

pid_act_parallel = Map(PassWID, scheduler=IPyClusterScheduler())

pid_act_parallel(inp=range(6))['out']

