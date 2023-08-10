import scipy.special as sp
import numpy as np
import xlwings as xl
import math

def my_transform(get_from,put_too):
    input_column = input_column=sht.range(get_from).options(expand='down').value
    my_min = min(input_column)
    my_max = max(input_column)
    myrange = my_max-my_min
    my_output = list(map(lambda x: (x-my_min)/myrange, input_column))
    sht.range(put_too).options(transpose=True).value=my_output

def shift_legendre(n,x):
    funct = math.sqrt(2*n+1) * sp.eval_sh_legendre(n,x)
    return funct

def do_slp(get_from,put_too,n):
    input_column=sht.range(get_from).options(expand='down').value
    slp = [shift_legendre(n, x) for x in input_column]
    sht.range(put_too).options(transpose=True).value=slp 
    

polynomial_orders = [1,2,3,4,5,6]
read_in = 'I4'
start_out_column = 25
start_out_row = 4

out_column = start_out_column
for p_order in polynomial_orders:
    output = (start_out_row, start_out_column)
    do_slp(read_in,output,p_order)
    start_out_column += 1
    



