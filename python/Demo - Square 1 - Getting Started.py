get_ipython().run_line_magic('run', "'Set-up.ipynb'")
get_ipython().run_line_magic('run', "'Loading scenes.ipynb'")

get_ipython().run_line_magic('run', "'vrep_models/PioneerP3DX.ipynb'")

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", '\n# Use the time library to set a wait duration\nimport time\n\n#Tell the robot to move forward by setting both motors to speed 1\nrobot.move_forward(1)\n\n#Wait for two seconds\ntime.sleep(2)\n\n#At the end of the programme the simulation stops\n#The robot returns to its original location')

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", '\n# Use the time library to set a wait duration\nimport time\n\n#YOUR CODE HERE')

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", "\n# Use the time library to set a wait duration\nimport time\n\n#YOUR CODE HERE\n\n#FOR EXAMPLE, TO DRIVE CLOCKWISE, USE: robot.rotate_right()\n#DON'T FORGET TO USE time.wait(TIME_IN_SECONDS) to give the robot time to turn")

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", 'import time\n\n#try to get the robot to draw an L shape: forward, right angle turn, forward\n')

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", 'import time\n\n#Program to draw a square')

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", 'import time\n\n#side 1\nrobot.move_forward()\ntime.sleep(1)\n#turn 1\nrobot.rotate_left(1.8)\ntime.sleep(0.45)\n#side 2\nrobot.move_forward()\ntime.sleep(1)\n#turn 2\nrobot.rotate_left(1.8)\ntime.sleep(0.45)\n#side 3\nrobot.move_forward()\ntime.sleep(1)\n#turn 3\nrobot.rotate_left(1.8)\ntime.sleep(0.45)\n#side 4\nrobot.move_forward()\ntime.sleep(1)')

