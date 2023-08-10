get_ipython().magic('reload_ext juno_magic')

get_ipython().magic('juno list')

get_ipython().magic('juno select "GBDX Testing"')

get_ipython().run_cell_magic('juno', '', '!ls /mnt/work/input/data')

