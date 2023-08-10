from __future__ import print_function
get_ipython().magic('matplotlib inline')

import sys
sys.path.append('../python')
import desc.sltimer

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

timer = desc.sltimer.SLTimer()

url = "https://raw.githubusercontent.com/COSMOGRAIL/PyCS/master/demo/demo1/data/trialcurves.txt"
timer.download(url, and_read=True)

timer.display_light_curves()

timer.estimate_time_delays(method='pycs', microlensing='spline', agn='spline', error=None, quietly=True)

timer.report_time_delays()

timer.display_light_curves()

# timer.estimate_uncertainties(n=1,npkl=4)

