get_ipython().magic('reload_ext pytriqs.magic')

get_ipython().run_cell_magic('triqs', '            ', 'int fun (int n) { \n   return n+1;\n}')

fun(2)



