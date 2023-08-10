#make

#write

from ctypes import *

#we call the method LoadLibrary and pass as argument a relative path to our library to load it to the python enviroment
#this line may be OS dependent, check for the slash or backslash directory navigation syntax
cdll.LoadLibrary("../fortran_library/bin/Trace_The_Envelope.so")
#the expected output is:
#>>> <CDLL '../fortran_library/bin/Trace_The_Envelope.so', handle 1e3b860 at 0x7fd098b9bc88>

#we use the method CDLL to create a pointer Trace_The_Envelope to that library
Trace_The_Envelope = CDLL("../fortran_library/bin/Trace_The_Envelope.so")

#we can call its functions as we would call functions in python packages
#attention should be paid to differences in underscores in the function names as in the src code and as in the compiling library, due the compiler actions.
_=Trace_The_Envelope.main_()

#we can even use ipython' timeit method to analyze the performance of that library
get_ipython().magic('timeit _=Trace_The_Envelope.main_()')
#(note that, as there are read/write commands on the fortran main function
#this will read/write the input/output files as many times as the profiler choose to loop the call)

import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

data = np.loadtxt("output/output.csv", delimiter=",", dtype=bytes, 
                  skiprows=1 #skip first row because the heading is incomplete in the csv
                 ).astype(str)

#see first line
print(data[0,:])

#see last line
print(data[-1,:])

#record the identification flags in variables for posterior usage
idLiq = data[0,0]
idVap = data[-1,0]

#keep column [0] in list as string, convert columns [1,2,3] to numpy array

numdata = (np.asarray((data[:,1:4]))).astype(np.float)

#see first line
print(numdata[0][:])

#see last line
print(numdata[-1][:])

# see that they are really understood as float
print(sum(sum(numdata))) #if they can be summed for example, they are really numbers

#split numdata into Pbub and Pdew according to the flags provided by the calculation program

Pbub = numdata[:,0][ np.where( data[:,0] == idLiq )]
Tbub = numdata[:,1][ np.where( data[:,0] == idLiq )]

Pdew = numdata[:,0][ np.where( data[:,0] == idVap )]
Tdew = numdata[:,1][ np.where( data[:,0] == idVap )]

#check
print(numdata[0:5,0])
print(data[0:5,0])
print(Pbub[0:5])

plt.plot(Tbub,Pbub)
plt.plot(Tdew,Pdew)

plt.xlabel('T') #use matplotlib's  mathtext to create special symbols in the x label
plt.ylabel('P') #y label for the second subplot

plt.show()

# GRAPHICAL ABSTRACT
get_ipython().magic('matplotlib inline')
from matplotlib import rcParams as rc

fig_width = 9 / 2.54 #in inches
fig_height = 9 / 2.54 #in inches
fig_size =  [fig_width,fig_height]

#FONTS & TICKS\n",
params = {'backend': 'ps',
'axes.labelsize': 12,  #in pts
'font.size': 8, #in pts
'legend.fontsize': 8, #in pts
'xtick.labelsize': 10, #in pts
'ytick.labelsize': 10, #in pts
'text.usetex': False,
'figure.figsize': fig_size}
rc.update(params)

GAfig, GAax1 = plt.subplots(1,1)

GAax1.plot(Tbub,Pbub*1e5)
GAax1.plot(Tdew,Pdew*1e5)


labels = [r'$P_{bub}(x_W)$', r'$P_{dew}(y_W)$']

plt.legend(labels, loc=2)

GAax1.set_title('[water; ethanol] \n LVE at ' + '351.55' + ' K')

GAax1.set_ylabel(r'$P(\mathrm{Pa})$')
GAax1.set_xlabel(r'$T(\mathrm{K})$')

GAax1.set_title('[0.3 (n-hexane); 0.7 (n-nonane)] \n' + r'$P \times T$' + ' phase envelope')

GAfig.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.14)

GAax1.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

GAfig.savefig('fig4.png', dpi=1000)

GAfig.show()

#min em T aqui: http://www.ddbst.com/en/EED/VLE/VLE%20Acetone%3BWater.php


