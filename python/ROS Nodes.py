get_ipython().run_cell_magic('bash', '--bg', 'roscore')

get_ipython().run_cell_magic('bash', '', 'rosnode list')

get_ipython().run_cell_magic('bash', '', 'rosnode info rosout')

get_ipython().run_cell_magic('bash', '--bg', 'rosrun turtlesim turtlesim_node')

get_ipython().run_cell_magic('bash', '', 'rosnode list')

get_ipython().run_cell_magic('bash', '--bg', 'rosrun turtlesim turtlesim_node __name:=my_turtle')

get_ipython().run_cell_magic('bash', '', 'rosnode list')

