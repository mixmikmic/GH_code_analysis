import numpy as np
import IEtools
import pylab as pl
get_ipython().magic('pylab inline')

filename1='C:/econdata/GDP.xls'
filename2='C:/econdata/PAYEMS.xls'
filename3='C:/econdata/CPIAUCSL.xls'

gdp = IEtools.FREDxlsRead(filename1)
lab = IEtools.FREDxlsRead(filename2)
cpi = IEtools.FREDxlsRead(filename3)

pl.plot(gdp['interp'].x,gdp['interp'](gdp['interp'].x))
pl.ylabel(gdp['name']+' [G$]')
pl.yscale('log')
pl.show()

pl.plot(gdp['growth'].x,gdp['growth'](gdp['growth'].x))
pl.ylabel(gdp['name']+' growth [%]')
pl.show()

result = IEtools.fitGeneralInfoEq(gdp['data'],lab['data'], guess=[1.0,0.0])
print(result)
print('IT index = ',np.round(result.x[0],decimals=2))
time=gdp['interp'].x
pl.plot(time,np.exp(result.x[0]*np.log(lab['interp'](time))+result.x[1]),label='model')
pl.plot(time,gdp['interp'](time),label='data')
pl.yscale('log')
pl.ylabel(gdp['name']+' [G$]')
pl.legend()
pl.show()

time=gdp['data'][:,0]

der1=gdp['growth'](time)-lab['growth'](time)
der2=cpi['growth'](time)
pl.plot(time,der1,label='model')
pl.plot(time,der2,label='data')
pl.legend()
pl.show()

time=gdp['data'][:,0]

der1=gdp['growth'](time)-cpi['growth'](time)
der2=lab['growth'](time)
pl.plot(time,der1,label='model')
pl.plot(time,der2,label='data')
pl.legend()
pl.show()



