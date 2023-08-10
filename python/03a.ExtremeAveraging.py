import numpy as np
import pandas as pd
import os

import sqlalchemy as sq
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

dbfile = 'huge.db'

try: os.remove(dbfile)
except: pass

Base = declarative_base()

class Number(Base):
    __tablename__ = 'number'
    
    id = sq.Column(sq.Integer, primary_key=True)
    value = sq.Column(sq.Float, nullable=False)

engine = sq.create_engine('sqlite:///'+dbfile)
Base.metadata.create_all(engine)

Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

N = int(1e3)

get_ipython().run_cell_magic('time', '', 'engine.execute(\n    Number.__table__.insert(),\n        [{"value": x} for x in xrange(N)]\n    )')

get_ipython().run_cell_magic('time', '', 'j = N\nwhile j < 2*N:\n    session.add(Number(value=float(j)))\n    session.commit()\n    j += 1')

N = int(1e10)
print np.round((N * ((20*1e-3) / 1e3) / 3600.0), 1),"hours"

N = int(1e7)

get_ipython().run_cell_magic('time', '', 'engine.execute(\n    Number.__table__.insert(),\n        [{"value": x} for x in xrange(2000, N)]\n    )')

get_ipython().system('du -h $dbfile')

0.5*(N-1)

def estimate_mean(method='sql_function'):
    
    import time as wallclock
    import guppy
    
    measure = guppy.hpy()
    
    start, end = {}, {}
    start['memory'] = measure.heap().size
    start['time'] = wallclock.time()
    
    df, mean = None, None

    if method == 'pandas_query':
        df = pd.read_sql(session.query(Number.value).statement, session.bind) 
        mean = np.mean(df.value)
        
    elif method == 'sql_function':
        mean = session.query(func.avg(Number.value)).one()[0]
        
    
    end['time'] = wallclock.time()
    end['memory'] = measure.heap().size

    time = (end['time']-start['time'])
    memory = (end['memory']-start['memory']) / (1024.0*1024.0)

    print "Estimated mean distance = ", mean

    print "Wallclock time spent = ", np.round(time,1), "seconds"
    print "Memory used = ",np.round(memory,1), "Mb"
    
    del df, mean

    return time, memory

t, m = estimate_mean(method='pandas_query')

t, m = estimate_mean(method='sql_function')

try: os.remove(dbfile)
except: pass

