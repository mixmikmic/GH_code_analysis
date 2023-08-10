get_ipython().run_line_magic('run', "'Set-up.ipynb'")
get_ipython().run_line_magic('run', "'Loading scenes.ipynb'")

#The following magic command allows us to embed dynamically created charts in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('run', "'vrep_models/lineTracer.ipynb'")

#Make use of time in demos
import time

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_grey lines.ttt' lineTracer", 'steps=10\nwhile steps:\n    robot.move_forward(2)\n    time.sleep(0.3)\n    steps=steps-1')

import pandas as pd
data={'light':pd.DataFrame()}

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_grey lines.ttt' lineTracer", "\nspeed=2\nsample_rate=0.3\nmax_rotations=5\n\nrobot.move_forward(speed)\nwhile robot.getrots()<max_rotations:    \n    data['light']=pd.concat([data['light'], pd.DataFrame([{'rots':robot.getrots(),\n                                                           'line_left':robot.left_line(),\n                                                           'light_left':robot.left_light()}])])\n    time.sleep(sample_rate)")

data['light'].plot(x='rots',y='light_left');



