import sys
if not '..' in sys.path:
    sys.path.insert(0, '..')
import control
import sympy
import numpy as np
import matplotlib.pyplot as plt
import ulog_tools as ut
import ulog_tools.control_opt as opt

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

log_file = ut.ulog.download_log('http://review.px4.io/download?log=35b27fdb-6a93-427a-b634-72ab45b9609e', '/tmp')
data = ut.sysid.prepare_data(log_file)
res = ut.sysid.attitude_sysid(data)
res

opt.attitude_loop_design(res['roll']['model'], 'ROLL', d)

attitude_loop_design(res['pitch']['model'], 'PITCH')

attitude_loop_design(res['yaw']['model'], 'YAW')

