# Loading Visualization

import plotly.offline as py
from plotly.graph_objs import *
import pandas as pd
import math
py.init_notebook_mode()

#we aim to visualize the following params: 
#'p1', 'p1est','p1meas', 'p2', 'p2est','p2meas', 'v', 'vest', 'yaw', 'yawest', 'yawrate', 'yawrateest', 'acc', 'yawacc'

#input file fields as they are saved into the UKF output file
my_cols=['time_stamp','p1est','p2est','vest','yawest','yawrateest','sens_type','NIS','p1meas','p2meas','p1','p2','v1_gt','v2_gt']
with open('../Unscented-Kalman-Filter/build/output.txt') as f:
    table_ekf_output = pd.read_table(f, sep='\t', names=my_cols, header=0, lineterminator='\n')
    

import math

v_col = []
v1 = table_ekf_output['v1_gt']
v2 = table_ekf_output['v2_gt']

for i in range(len(v1)):
    v_col.append(math.sqrt(v1[i]**2+v2[i]**2))
    
table_ekf_output['v'] = v_col

NIS_laser = []
NIS_radar = []
sensor = table_ekf_output['sens_type']
NIS = table_ekf_output['NIS']

for i in range(len(sensor)):
    if sensor[i] == 'lidar':
        NIS_laser.append(NIS[i])
        NIS_radar.append(0.0)
    else:
        NIS_laser.append(0.0)
        NIS_radar.append(NIS[i])
        
table_ekf_output['NIS_laser'] = NIS_laser
table_ekf_output['NIS_radar'] = NIS_radar

#check the parsed file
table_ekf_output[0:5]

import plotly.offline as py
from plotly.graph_objs import *




#Ground Truth
trace1 = Scatter(
    x=table_ekf_output['p1'],
    y=table_ekf_output['p2'],
    xaxis='x2',
    yaxis='y2',
    name = 'ground truth position',
    mode = 'markers'      
)


#estimations
trace2 = Scatter(
    x=table_ekf_output['p1est'],
    y=table_ekf_output['p2est'],
    xaxis='x2',
    yaxis='y2',
    name='UKF position estimation',
    mode = 'markers'       
)

#Measurements
trace3 = Scatter(
    x=table_ekf_output['p1meas'],
    y=table_ekf_output['p2meas'],
    xaxis='x2',
    yaxis='y2',
    name = 'position measurements',
    #mode = 'markers'
)


data = [trace1, trace2, trace3]

layout = Layout(
    xaxis2=dict(
   
        anchor='x2',
        title='px in m'
    ),
    yaxis2=dict(
    
        anchor='y2',
        title='py in m'
    )
)

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename= 'EKF')

import plotly.offline as py
from plotly.graph_objs import *

t = range(0,len(table_ekf_output))
## we want: 'p1', 'p1est','p1meas', 'p2', 'p2est','p2meas', 'v', 'vest', 'yaw', 'yawest', 'yawrate', 'yawrateest', 'acc', 'yawacc'
#['p1est','p2est','vest','yawest','yawrateest','p1meas','p2meas','p1','p2','v','yaw', 'yawrate','v1_gt','v2_gt']


trace1 = Scatter(
    x=t,
    y=table_ekf_output['p1'],
    xaxis='x2',
    yaxis='y2',
    name = 'px ground truth in m',
    #mode = 'markers'
)

trace2= Scatter(
    x=t,
    y=table_ekf_output['p1est'],
    xaxis='x2',
    yaxis='y2',
    name = 'px estimated in m ',
    #mode = 'markers'
)

trace3 = Scatter(
    x=t,
    y=table_ekf_output['p1meas'],
    xaxis='x2',
    yaxis='y2',
    name = 'px measured in m ',
    #mode = 'markers'

)



data = [trace1, trace2, trace3]

layout = Layout(
    xaxis2=dict(
   
        anchor='x2',
        title='t'
    ),
    yaxis2=dict(
    
        anchor='y2',
        title='m'
    )
)

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename= 'EKF')

import plotly.offline as py
from plotly.graph_objs import *

t = range(0,len(table_ekf_output))
## we want: 'p1', 'p1est','p1meas', 'p2', 'p2est','p2meas', 'v', 'vest', 'yaw', 'yawest', 'yawrate', 'yawrateest', 'acc', 'yawacc'
#['p1est','p2est','vest','yawest','yawrateest','p1meas','p2meas','p1','p2','v','yaw', 'yawrate','v1_gt','v2_gt']


trace7 = Scatter(
    x=t,
    y=table_ekf_output['v'],
    xaxis='x2',
    yaxis='y2',
    name = 'ground truth velocity ',
    #mode = 'markers'
)

trace8 = Scatter(
    x=t,
    y=table_ekf_output['vest'],
    xaxis='x2',
    yaxis='y2',
    name = 'estimated velocity ',
    #mode = 'markers'
)



data = [trace7, trace8]

layout = Layout(
    xaxis2=dict(
   
        anchor='x2',
        title='t'
    ),
    yaxis2=dict(
    
        anchor='y2',
        title='m/s'
    )
)

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename= 'EKF')

import plotly.offline as py
from plotly.graph_objs import *

t = range(0,len(table_ekf_output))
## we want: 'p1', 'p1est','p1meas', 'p2', 'p2est','p2meas', 'v', 'vest', 'yaw', 'yawest', 'yawrate', 'yawrateest', 'acc', 'yawacc'
#['p1est','p2est','vest','yawest','yawrateest','p1meas','p2meas','p1','p2','v','yaw', 'yawrate','v1_gt','v2_gt']


trace7 = Scatter(
    x=t,
    y=table_ekf_output['yaw'],
    xaxis='x2',
    yaxis='y2',
    name = 'ground truth yaw angle ',
    #mode = 'markers'
)

trace8 = Scatter(
    x=t,
    y=table_ekf_output['yawest'],
    xaxis='x2',
    yaxis='y2',
    name = 'estimated yaw angle ',
    #mode = 'markers'
)



data = [trace7, trace8]

layout = Layout(
    xaxis2=dict(
   
        anchor='x2',
        title='t'
    ),
    yaxis2=dict(
    
        anchor='y2',
        title='rad'
    )
)

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename= 'EKF')

import plotly.offline as py
from plotly.graph_objs import *

t = range(0,len(table_ekf_output))
## we want: 'p1', 'p1est','p1meas', 'p2', 'p2est','p2meas', 'v', 'vest', 'yaw', 'yawest', 'yawrate', 'yawrateest', 'acc', 'yawacc'
#['p1est','p2est','vest','yawest','yawrateest','p1meas','p2meas','p1','p2','v','yaw', 'yawrate','v1_gt','v2_gt']


trace7 = Scatter(
    x=t,
    y=table_ekf_output['yawrate'],
    xaxis='x2',
    yaxis='y2',
    name = 'ground truth yaw rate ',
    #mode = 'markers'
)

trace8 = Scatter(
    x=t,
    y=table_ekf_output['yawrateest'],
    xaxis='x2',
    yaxis='y2',
    name = 'estimated yaw rate',
    #mode = 'markers'
)



data = [trace7, trace8]

layout = Layout(
    xaxis2=dict(
   
        anchor='x2',
        title='t'
    ),
    yaxis2=dict(
    
        anchor='y2',
        title='rad/s'
    )
)

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename= 'EKF')

import plotly.offline as py
from plotly.graph_objs import *

t = range(0,len(table_ekf_output))
## we want: 'p1', 'p1est','p1meas', 'p2', 'p2est','p2meas', 'v', 'vest', 'yaw', 'yawest', 'yawrate', 'yawrateest', 'acc', 'yawacc'
#['p1est','p2est','vest','yawest','yawrateest','p1meas','p2meas','p1','p2','v','yaw', 'yawrate','v1_gt','v2_gt']


trace2 = Scatter(
    x=t,
    y=table_ekf_output['NIS_laser'],
    xaxis='x2',
    yaxis='y2',
    name = 'NIS laser',
    #mode = 'markers'
)


trace1= Scatter(
    x=[t[0], t[-1]],
    y=[5.991, 5.991],
    xaxis='x2',
    yaxis='y2',
    name = '95 %',
    #mode = 'markers'
)



data = [trace1, trace2]

layout = Layout(
    xaxis2=dict(
   
        anchor='x2',
        title='k'
    ),
    yaxis2=dict(
    
        anchor='y2',
        #title='py'
    )
)

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename= 'EKF')

import plotly.offline as py
from plotly.graph_objs import *

t = range(0,len(table_ekf_output))
## we want: 'p1', 'p1est','p1meas', 'p2', 'p2est','p2meas', 'v', 'vest', 'yaw', 'yawest', 'yawrate', 'yawrateest', 'acc', 'yawacc'
#['p1est','p2est','vest','yawest','yawrateest','p1meas','p2meas','p1','p2','v','yaw', 'yawrate','v1_gt','v2_gt']


trace2 = Scatter(
    x=t,
    y=table_ekf_output['NIS_radar'],
    xaxis='x2',
    yaxis='y2',
    name = 'NIS radar',
    #mode = 'markers'
)


trace1= Scatter(
    x=[t[0], t[-1]],
    y=[7.815, 7.815],
    xaxis='x2',
    yaxis='y2',
    name = '95 %',
    #mode = 'markers'
)



data = [trace1, trace2]

layout = Layout(
    xaxis2=dict(
   
        anchor='x2',
        title='k'
    ),
    yaxis2=dict(
    
        anchor='y2',
        #title='py'
    )
)

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename= 'EKF')

import plotly.offline as py
from plotly.graph_objs import *

t = range(0,len(table_ekf_output))
## we want: 'p1', 'p1est','p1meas', 'p2', 'p2est','p2meas', 'v', 'vest', 'yaw', 'yawest', 'yawrate', 'yawrateest', 'acc', 'yawacc'
#['p1est','p2est','vest','yawest','yawrateest','p1meas','p2meas','p1','p2','v','yaw', 'yawrate','v1_gt','v2_gt']


trace1 = Scatter(
    x=t,
    y=table_ekf_output['p1'],
    xaxis='x2',
    yaxis='y2',
    name = 'p1',
    #mode = 'markers'
)

trace2= Scatter(
    x=t,
    y=table_ekf_output['p1est'],
    xaxis='x2',
    yaxis='y2',
    name = 'p1est',
    #mode = 'markers'
)

trace3 = Scatter(
    x=t,
    y=table_ekf_output['p1meas'],
    xaxis='x2',
    yaxis='y2',
    name = 'p1meas',
    #mode = 'markers'
)

trace4= Scatter(
    x=t,
    y=table_ekf_output['p2'],
    xaxis='x2',
    yaxis='y2',
    name = 'p2',
    #mode = 'markers'
)

trace5 = Scatter(
    x=t,
    y=table_ekf_output['p2est'],
    xaxis='x2',
    yaxis='y2',
    name = 'p2est',
    #mode = 'markers'
)

trace6= Scatter(
    x=t,
    y=table_ekf_output['p2meas'],
    xaxis='x2',
    yaxis='y2',
    name = 'p2meas',
    #mode = 'markers'
)

trace7 = Scatter(
    x=t,
    y=table_ekf_output['v'],
    xaxis='x2',
    yaxis='y2',
    name = 'v',
    #mode = 'markers'
)

trace8 = Scatter(
    x=t,
    y=table_ekf_output['vest'],
    xaxis='x2',
    yaxis='y2',
    name = 'vest',
    #mode = 'markers'
)

trace9= Scatter(
    x=t,
    y=table_ekf_output['yaw'],
    xaxis='x2',
    yaxis='y2',
    name = 'yaw',
    #mode = 'markers'
)

trace10 = Scatter(
    x=t,
    y=table_ekf_output['yawest'],
    xaxis='x2',
    yaxis='y2',
    name = 'yawest',
    #mode = 'markers'
)

trace11= Scatter(
    x=t,
    y=table_ekf_output['yawrate'],
    xaxis='x2',
    yaxis='y2',
    name = 'yawrate',
    #mode = 'markers'
)

trace12= Scatter(
    x=t,
    y=table_ekf_output['yawrateest'],
    xaxis='x2',
    yaxis='y2',
    name = 'yawrateest',
    #mode = 'markers'
)

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12]

layout = Layout(
    xaxis2=dict(
   
        anchor='x2',
        title='t'
    ),
    yaxis2=dict(
    
        anchor='y2',
        #title='py'
    )
)

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename= 'EKF')





