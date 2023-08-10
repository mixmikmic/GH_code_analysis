CURRENT_SAMPLE="S1W7D21_15_09_11"
# import helper functions including automated setup
from agilentpyvisa.reram_helpers_B1500 import *

# display images in the notebook %matplotlib notebook makes them interactive!
get_ipython().magic('matplotlib inline')

import os
import os.path
today=datetime.today().strftime("%Y-%m-%d")
already_in = today in os.path.abspath(".")
if not already_in and not os.path.exists(today):
    os.mkdir(today)
if not already_in:
    os.chdir(today)
print("Working directory:")
print(os.path.abspath("."))

form_data={}
annealing_data={}

form_sweep= plt.figure(figsize=[10,5])
f=form(3,100,10e-3, mrange=MeasureRanges_I.uA10_limited,gate=1.9)
f.to_csv("{}_form_3V.csv".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
plt.autoscale()
form_data={}
form_data['FORM_GATE']=1.9
print(checkR(CURRENT_SAMPLE))
form_data['FORM_V']=find_set_V(f)
print("Forming Voltage:",form_data['FORM_V'])

print(checkR(CURRENT_SAMPLE))
r=reset_sweep(-1.5,100,5e-3, mrange=MeasureRanges_I.uA10_limited,gate=1.9,plot=True)
r.to_csv("{}_reset_{}.csv".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), CURRENT_SAMPLE))
print(checkR(CURRENT_SAMPLE))

print(checkR(CURRENT_SAMPLE))
s=set_sweep(1.0,100,5e-3, mrange=MeasureRanges_I.uA10_limited,gate=1.9,plot=True)
s.to_csv("{}_set_{}.csv".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), CURRENT_SAMPLE))
print("Set Voltage:", find_set_V(s))
print(checkR(CURRENT_SAMPLE))

rh=get_hist(r)
plt.plot(rh['EV'],rh['R'])

sh=get_hist(s)
plt.plot(sh['EV'],sh['R'])

frames,annealing_data=anneal(CURRENT_SAMPLE,setV=1.0,resetV=-1.5,gateV=1.9,steps=100,times=3, plot=True,sleep_between=1)

pat = get_pyramid_pattern

