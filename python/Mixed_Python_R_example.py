import numpy
get_ipython().magic('load_ext rpy2.ipython')

x=numpy.random.randn(100)
beta=3
y=beta*x+numpy.random.randn(100)

get_ipython().run_cell_magic('R', '-i x,y -o beta_est', 'result=lm(y~x)\nbeta_est=result$coefficients\nsummary(result)')

print(beta_est)



