get_ipython().magic('pylab inline')

import ptha_paths

from ipywidgets import interact
from IPython.display import Image, display
import sys,os

import hazard_maps_functions as HM
import hazard_curve_functions as HC
import hazard_quiz as Q

p1 = 0.001
h1 = 1.5
E1 = [p1,h1]
event_list = [E1]  # only one event in this example
h_ev,p_ev = HC.hazard_curve(event_list, 3)
fig = HC.plot_hazard_curve(h_ev,p_ev)

E2 = [0.003, 1.0]      # the second event
event_list = [E1, E2]  # assumes E1 already defined in previous cell
h_ev,p_ev = HC.hazard_curve(event_list, 3)
fig = HC.plot_hazard_curve(h_ev,p_ev)

p1 = 0.1
p2 = 0.6

p_hat = p1 + p2 - p1*p2
print "When p1 = %g and p2 = %g, the probability of at least one event is p_hat = %g" % (p1,p2,p_hat)

interact(HC.makefig, p2=[0.0005, 0.005, 0.0005], h2=[0.,2.5,0.5], )

p_hat = "?"
Q.check_answer1(p_hat)

from numpy import random
random.seed(12345)  # seed the random number generator

event_list = []  # initialize to empty list
n_events = 10
random_h = 0.5 + 2*rand(n_events)  # rand(n) returns array of n random number in interval [0,1]
print "Flooding depths h_k = ",random_h

for k in range(n_events):
    pk = 0.002
    hk = random_h[k]
    Ek = [pk,hk]
    event_list.append(Ek)   # add this event to the list
h_ev,p_ev = HC.hazard_curve(event_list, 3)
fig = HC.plot_hazard_curve(h_ev,p_ev)

p_hat = "?"
Q.check_answer2(p_hat)

def show_hc(k):
    display(Image(HM.hc_split_plots[k],width=500))
    
interact(show_hc,k=(0,len(HM.hc_split_plots)-1))

