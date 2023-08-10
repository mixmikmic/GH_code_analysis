import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

# inputs
R = 1000  # resistace in ohms
C = 0.001  # capacitance in farads

tau = R*C
fc = 2*np.pi*tau
print('Time constant, tau =', tau, 's')
print('Cutoff frequency, f =', fc, 'Hz')

# inputs
Vi = 5  # volts

# define a linear space of omega that is significantly greatr that fc
omega = np.linspace(0.01*fc,fc*5,1000)
Vo_lp = Vi*1/np.sqrt(1+(omega*tau)**2)

Gdb_lp = 20*np.log10(Vo_lp/Vi)  # Where Gdb is the power 

# plot with plotly
# Create traces
legend = ['Low Pass Gain']
tracelp = go.Scatter(
    x=np.log10(omega),
    y=Gdb_lp,
    mode='lines',
    name=legend[0]
)

# Edit the layout
layout = dict(title='Output Voltage of First Order Low-Pass Filter vs. Time',
              xaxis=dict(title='Log[Frequency (Hz)]'),
              yaxis=dict(title='Power Gain (dB)'),
              )
data = [tracelp]  # put trace in array (plotly formatting)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename="FirstOrderLPFilter")

freq3db = 0 # logarithmic frequency at -3db
freqcut = 10**freq3db
print('Therefore, the cutoff frequency is', freqcut, 'Hz')

Vo_hp = Vi*(omega*tau)/np.sqrt(1+(omega*tau)**2)

Gdb_hp = 20*np.log10(Vo_hp/Vi)  # Where Gdb is the power 

# plot with plotly
# Create traces
legend = ['High Pass Gain']
tracehp = go.Scatter(
    x=np.log10(omega),
    y=Gdb_hp,
    mode='lines',
    name=legend[0]
)

# Edit the layout
layout = dict(title='Output Voltage of First Order High-Pass Filter vs. Time',
              xaxis=dict(title='Log[Frequency (Hz)]'),
              yaxis=dict(title='Power Gain (dB)'),
              )
data = [tracehp]  # put trace in array (plotly formatting)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename="FirstOrderHPFilter")

# Edit the layout
layout = dict(title='Output Voltage of First Order High-Pass and Low-Pass Filter vs. Time',
              xaxis=dict(title='Log[Frequency (Hz)]'),
              yaxis=dict(title='Power Gain (dB)'),
              )
data = [tracehp, tracelp]  # put trace in array (plotly formatting)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename="FirstOrderFilters")

