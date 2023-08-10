from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

file = 'Conv_S-1_#3.txt' # C:\Users\Nicolas\PycharmProjects\untitled\Dataset\Corriendo\Conv_s-1_#3.txt (using absolute route)
list = [line.rstrip('\n') for line in open(file)]
data = [int(x) for x in list] # Parse String to int
print (data)

maximo = np.max(data)
minimo = np.min(data)
media  = np.mean(data)
varianza = np.var(data)

a = "Maximo:",maximo
b = "Minimo:",minimo
c = "Media:", media
d = "Varianza:",varianza

print ("Maximo:",maximo)
print ("Minimo:",minimo)
print ("Media:", media)
print("Varianza:",varianza)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_title(file)
ax1.set_xlabel('Number of Samples')
ax1.set_ylabel('Acelometer analog value')
ax1.plot(data, c='b', label=a+b+c+d)

leg = ax1.legend()

plt.show()

from sklearn import preprocessing
import numpy as np

data_fit = np.array(data).reshape((len(data), -1))
print(data_fit)

scaler = preprocessing.StandardScaler().fit(data_fit)
print('Media Scalar API:',scaler.mean_)
print('Scala API:', scaler.scale_)
print('Transform API:', scaler.transform(data_fit))

from scipy.fftpack import fft, rfft, irfft
import matplotlib.pyplot as plt

file = 'Conv_S-1_#3.txt' 
list = [line.rstrip('\n') for line in open(file)]
data = [int(x) for x in list] # Parse String to int

fft(data)
print (fft(data))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Trasnformada de Fourier")
ax1.set_xlabel('Numero de muestras')
ax1.set_ylabel('Muestras Sensor')
ax1.plot(fft(data), c='b', label='Trasnformada de Fourier')
leg = ax1.legend()
plt.show()

yr = rfft(data)
print (yr)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Coeficientes de Fourier")
ax1.set_xlabel('Numero de muestras')
ax1.set_ylabel('Muestras Sensor')
ax1.plot(yr, c='b', label='Coeficientes de Fourier')
leg = ax1.legend()
plt.show()



