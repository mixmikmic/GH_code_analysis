import time

def myFunction():
    print("Hello...")
    
    #Pause awhile...
    time.sleep(2)
    
    print("...world!")
    
#call the function - note the brackets!
myFunction()

get_ipython().run_line_magic('run', "'Set-up.ipynb'")
get_ipython().run_line_magic('run', "'Loading scenes.ipynb'")
get_ipython().run_line_magic('run', "'vrep_models/PioneerP3DX.ipynb'")

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", '\n#Your code  - using functions - here')

get_ipython().run_cell_magic('vrepsim', "'../scenes/OU_Pioneer.ttt' PioneerP3DX", 'import time\n\nside_speed=2\nside_length_time=1\nturn_speed=1.8\nturn_time=0.45\n\nnumber_of_sides=4\n\ndef traverse_side():\n     \n    pass\ndef turn():\n    pass\n    \nfor side in range(number_of_sides):\n        \n    #side\n    robot.move_forward(side_speed)\n    time.sleep(side_length_time) \n    \n    #turn\n    robot.rotate_left(turn_speed)\n    time.sleep(turn_time)')

