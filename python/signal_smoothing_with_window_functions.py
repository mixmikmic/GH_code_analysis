get_ipython().magic("run 'database_connectivity_setup.ipynb'")

get_ipython().magic('matplotlib inline')
import numpy as np
import math as math
import seaborn as sns
import pylab
from matplotlib import pyplot as plt

ts = np.arange(-2*math.pi,2*math.pi,step=0.1)
sigma = 0.1+0.3*np.random.uniform(0,1,1)
f = np.sin(ts) + np.random.normal(0,sigma,len(ts))
plt.plot(ts,f)
plt.show()

sql = """
    DROP TABLE IF EXISTS iot.conv_sensor_input;
"""
psql.execute(sql,conn)

sql = """
    CREATE TABLE iot.conv_sensor_input (        
        run_number int,
        ts float8,
        f float8
    ) DISTRIBUTED BY (run_number);
"""
psql.execute(sql,conn)
conn.commit()

ts = np.arange(-2*math.pi,2*math.pi,step=0.1)
sigma = 0.1+0.3*np.random.uniform(0,1,1)
f = np.sin(ts) + np.random.normal(0,sigma,len(ts))
for j in np.arange(0,len(ts)):
    sql = """
        INSERT INTO iot.conv_sensor_input VALUES (
            {run_number},
            {tp},
            {fp}
        );
    """.format(
            run_number=str(1),
            tp = str(ts[j]),
            fp = str(f[j])
        )
    psql.execute(sql,conn)
conn.commit()

sql = """
    SELECT
        run_number,
        array_agg(ts ORDER BY ts) as ts_arr,
        array_agg(f ORDER BY ts) as f_arr,
        array_agg(fs ORDER BY ts) as fs_arr
    FROM (
        SELECT
            run_number,
            ts,
            f,
            (
                f + 
                lead(f,1) OVER (PARTITION BY run_number ORDER BY ts) +
                lag(f,1) OVER (PARTITION BY run_number ORDER BY ts)
            )/3.0 as fs
        FROM
            iot.conv_sensor_input
        ORDER BY ts
    ) q
    WHERE run_number = 1
    GROUP BY 1
"""
df = psql.read_sql(sql,conn)
time = np.array(df.loc[0]['ts_arr'])
signal = np.array(df.loc[0]['f_arr'])
smoothed_signal = np.array(df.loc[0]['fs_arr'])
pylab.plot(time,signal,color='r',label='Noisy input')
pylab.plot(time,smoothed_signal,color='b', label='Smoothed output')
pylab.legend(loc='upper left')
plt.show()

conn.close()

