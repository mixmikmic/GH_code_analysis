import numpy as np
import json

# import simpy.rt # Uses 'real time' wall clock time. This won't work in a Jupyter notebook.
# env = simpy.rt.RealtimeEnvironment()

import simpy
env = simpy.Environment()

# Simulation parameters

NUM_RECIPIENTS = 10

SIM_DURATION = 1000

MIN_QUEUE_TIME = 1
MAX_QUEUE_TIME = 10

MIN_DELIVERY_TIME = 1
MAX_DELIVERY_TIME = 50

MIN_READ_TIME = 5
MAX_READ_TIME = 600

def get_outdict(time, userid, status):
    return {'time': time, 'userid': userid, 'Status': status}

def message_status(env, userid):
    outdict = get_outdict(env.now, userid, 'Queued')
    print(json.dumps(outdict))
    
    yield env.timeout(np.random.uniform(low=MIN_QUEUE_TIME,
                                        high=MAX_QUEUE_TIME))

    outdict = get_outdict(env.now, userid, 'Sent')
    print(json.dumps(outdict))
    
    yield env.timeout(np.random.uniform(low=MIN_DELIVERY_TIME,
                                        high=MAX_DELIVERY_TIME))

    outdict = get_outdict(env.now, userid, 'Delivered')
    print(json.dumps(outdict))
    
    yield env.timeout(np.random.uniform(low=MIN_READ_TIME,
                                        high=MAX_READ_TIME))

    outdict = get_outdict(env.now, userid, 'Read')
    print(json.dumps(outdict))

# Setup and start the simulation

for userid in range(1, NUM_RECIPIENTS):
    message = message_status(env, userid)
    env.process(message)

env.run(until=SIM_DURATION)



