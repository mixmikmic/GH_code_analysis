get_ipython().run_line_magic('run', "'Set-up.ipynb'")
get_ipython().run_line_magic('run', "'Loading scenes.ipynb'")
get_ipython().run_line_magic('run', "'vrep_models/PioneerP3DX.ipynb'")

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", '\n# Use the time library to set a wait duration\nimport time\n\n#Tell the robot to move forward by setting both motors to speed 1\nrobot.move_forward(1)\nprint(robot.getvalleft(), robot.getvalright())\n#Wait for two seconds\ntime.sleep(2)\nprint(robot.getvalleft(), robot.getvalright())\nrobot.rotate_left()\ntime.sleep(2)\nprint(robot.getvalleft(), robot.getvalright())\n#At the end of the programme the simulation stops\n#The robot returns to its original location')

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", '\n\nget_orientation_degrees')

#need to add the odometry/rotation count child script to the robot model

