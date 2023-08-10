from wowp.actors import FuncActor
from wowp.schedulers import LinearizedScheduler, IPyClusterScheduler
from wowp.actors.mapreduce import Map
from wowp.actors.experimental import Chain
from wowp.util import ConstructorWrapper
from wowp.actors.mapreduce import PassWID

# numpy will perform some calculations
import numpy as np

Sin = ConstructorWrapper(FuncActor, np.sin)
Annotate = ConstructorWrapper(PassWID)

chain = Chain('chain', (Sin, Annotate))

chain(inp=np.pi/2)['out']

map_parallel = Map(
    Chain,
    args=('chain', (Sin, Annotate), ),
    scheduler=IPyClusterScheduler())

inp = np.linspace(0, np.pi, 10)
results_p = map_parallel(inp=inp)

results_p['out']

