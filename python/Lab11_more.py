get_ipython().magic('matplotlib inline')
import numpy as np

weather = np.loadtxt('weather_space.txt')
print(weather)
print(type(weather) )
print(weather.shape)

## weather_space.txt now only read in column 1 and 2 when you add keyword usecols = [1,2],
weather1 = np.loadtxt('weather_space.txt', usecols = [1,2])
print(weather1)
print(type(weather1) )
print(weather1.shape,'\n')

## weather_space.txt now only read in column 0, when you add keyword usecols = [0],
## But now, the result is just a 1-d array
weather1 = np.loadtxt('weather_space.txt', usecols = [0])
print(weather1)
print(type(weather1) )
print(weather1.shape)

# weather = np.loadtxt('weather_comma.txt')# does not work
weather = np.loadtxt('weather_comma.txt',delimiter=',')# does not work
print(weather)
print(type(weather) )
print(weather.shape)

#weather = np.loadtxt('funny_weather.txt',delimiter=',')# does not work because there are strings

weather = np.loadtxt('funny_weather.txt',delimiter=',',usecols=[1,2,3])
print(weather)
print(type(weather) )
print(weather.shape)

weather = np.loadtxt('funny_weather.txt',delimiter=',',usecols=[1,2,3],skiprows=2) # skip the top two rows
print(weather)
print(type(weather) )
print(weather.shape)

import os

## store current working directory
cwd      = os.getcwd() 


## a universal standard is to store all data files in a subfolder 
## called "data" or "raw_data" or "blabla_data"

datapath = cwd + u'/data/'

## You cannot access houston_weather.txt simply by np.loadtxt('houston_weather.txt', delimiter=',')
houston_weather = np.loadtxt(datapath+'houston_weather.txt', delimiter=',')
weather = np.loadtxt(datapath + 'weather_space.txt')

weather = np.loadtxt(datapath + 'weather_space.txt')

print("This is NOT a matrix product, but entrywise squared!!!\n", weather**2)
print("\nThis is NOT a matrix product, but entrywise product!!!\n", weather*weather)
print("\nThis is NOT a matrix inverse, but entrywise inverse!!!\n", weather**(-1))

P = np.matrix(weather)
print("true matrix product: P times P\n", P**2)
print("\nweather times weather\n", weather**2)



