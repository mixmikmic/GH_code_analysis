get_ipython().magic('pylab')
import pysd
from matplotlib import animation
import numpy as np

#import the model (need to import each time to reinitialize) 
#choose one of the following lines:
#model = pysd.read_vensim('../../models/Basic_Structures/First_Order_Delay.mdl')
model = pysd.read_vensim('../../models/Basic_Structures/Third_Order_Delay.mdl')

#set the delay time in the model
model.set_components({'delay':5})

#set the animation parameters
fps=4
seconds=60
dt=1./fps

#set up the figure axes
fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(-10, 20)
ax.set_xticks([])
title = ax.set_title('Time %.02f'%0)

#draw the target line
ax.plot([0,1], [10,10], 'r--')

#draw the moving line, just for now. We'll change it later
line, = ax.plot([0,1], [0,0], lw=2)

#set up variables for simulation
input_val = 1
model.components.input = lambda: input_val

#capture keyboard input
def on_key_press(event):
    global input_val
    if event.key == 'up':
        input_val += .25
    elif event.key == 'down':
        input_val -= .25
    sys.stdout.flush()
    
fig.canvas.mpl_connect('key_press_event', on_key_press)

#make the animation
def animate(t):
    #run the simulation forward
    time = model.components.t+dt
    stocks = model.run(return_columns=['input', 'delay_buffer_1', 'delay_buffer_2', 'delay_buffer_3', 'output'],
                       return_timestamps=[time], 
                       initial_condition='current', collect=True)
 
    #make changes to the display
    level = stocks['output']
    line.set_data([0,1], [level, level])
    title.set_text('Time %.02f'%time)
    
# call the animator.  
anim = animation.FuncAnimation(fig, animate, repeat=False,
                               frames=seconds*fps, interval=1000./fps, 
                               blit=False)



record = model.get_record()
record.head()

record.plot();

plt.plot(x,input_collector, label='Your Input')
plt.plot(x,y, label='Model Response')
plt.legend(loc='lower right')
plt.xlabel('Time [Seconds]')
plt.ylabel('Value');

import pandas as pd
delay_stock_values = pd.DataFrame(stocks_collector)
delay_stock_values.plot()
plt.xlabel('Time [Seconds]')
plt.ylabel('Stock Level');



