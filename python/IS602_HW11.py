import Tkinter
import tkFileDialog
import pandas as pd
import numpy as np
from pandas import Series

root = Tkinter.Tk()
root.withdraw()
file_p = tkFileDialog.askopenfilename(parent=root) # select the apple data file
AD = pd.read_csv(file_p)
AD.columns = ['date','price','p_change']

print list(AD.columns.values) #checking to make sure the headers are correct
print AD.shape # find the dimensions of the df

p = Series(AD['p_change'][1:],dtype=float) # not using first row (headers)
mu, sigma = np.mean(p), np.std(p)
mu, sigma

f_price = []
for i in range(0,10000):
    l_price = AD['price'][251] # the last row (we aren't looking at the headers)
    diff_20 = np.random.normal(mu, sigma, 20)
    next_20 = [] # nesting to keep it simple
    for e in diff_20:
        l_price = l_price + l_price * e #notation shared by David Stern
        next_20.append(l_price)
    f_price.append(next_20[19])

print np.percentile(f_price,1) 

