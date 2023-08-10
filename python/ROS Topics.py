get_ipython().run_cell_magic('bash', '', 'rosnode list')

get_ipython().run_cell_magic('bash', '--bg', 'rosrun turtlesim turtlesim_node')

get_ipython().run_cell_magic('bash', '', 'rostopic -h')

get_ipython().run_cell_magic('bash', '', 'rostopic list -v')

get_ipython().run_cell_magic('bash', '', 'rostopic type /turtle1/cmd_vel')

get_ipython().run_cell_magic('bash', '', 'rosmsg show geometry_msgs/Twist')

get_ipython().run_cell_magic('bash', '', "rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- \\\n'[2.0, 0.0, 0.0]' '[0.0, 0.0, 1.8]'")

