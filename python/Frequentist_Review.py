from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from ipywidgets import interact, interactive, fixed, interact_manual
get_ipython().run_line_magic('matplotlib', 'inline')

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

def free_fall_variance(y0, number_measurements, sigma):
    
    g = 9.81
    t_fall = np.sqrt(2*y0/g)
    mu = t_fall
    t_measured = np.random.normal(mu, sigma, number_measurements)
    t_measured = np.round(t_measured, 2)
    
    return t_measured


def f(number_measurements):
    
    t_measured = free_fall_variance(20, number_measurements, 0.25)
    binsize = np.int(np.round(np.sqrt(len(t_measured)))/(np.max(t_measured)-np.min(t_measured)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(t_measured, edgecolor = 'black', bins = 20)
    ax.set_xlim([1, 3])
    ax.set_xlabel('Drop time (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Ball Drop Times')
    plt.show()
    
    return 
interact(f, number_measurements = (5, 1000))
plt.show()
        
        


t_measured_fixed = free_fall_variance(20, 1000, 0.25)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(t_measured_fixed, edgecolor = 'black', bins = 20)
ax.set_xlim([1, 3])
ax.set_ylim([0, 175])
ax.set_xlabel('Drop time (seconds)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Ball Drop Times')
plt.show()

def f(standard_deviation):
    
    t_measured = free_fall_variance(20, 1000, standard_deviation)
    binsize = np.int(np.round(np.sqrt(len(t_measured)))/(np.max(t_measured)-np.min(t_measured)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(t_measured, edgecolor = 'black', bins = 25)
    ax.set_xlim([0, 4])
    ax.set_xlabel('Drop time (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Ball Drop Times')
    ax.set_ylim([0, 225])
    plt.show()
    
    return 
interact(f, standard_deviation = (0.02, 0.75))
plt.show()

droptimes = np.zeros((15, 7))
distances = np.arange(20, 55, 5)

for d in range(len(distances)):
    
    droptimes[:, d] = free_fall_variance(distances[d], 15, 0.20)
    

    
print(droptimes)
names = ['Distance: 35', 'Distance: 30', 'Distance: 25', 'Distance: 20', 'Distance: 15', 'Distance: 10', 'Distance: 5']

np.vstack((names, droptimes))
np.savetxt('fall_1.csv', droptimes, fmt='%.18e', delimiter=' ', newline='\n')


print(t1)
        





