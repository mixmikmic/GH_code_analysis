get_ipython().system('rm output.txt filter.txt')

get_ipython().magic('pylab inline')

import subprocess

def NoveltyFilter(path,prime,K,N,mapName,inputName,outputName,filterName,intervall):
    print "{0}/NoveltyFilter".format(path) +" "+str(prime)+" "+str(K)+" "+str(N)+" "+mapName+" "+inputName+" "+outputName+" "+filterName+" "+str(intervall)
    res=subprocess.check_output(["./a.out",str(prime),str(K),str(N),path+"/"+mapName,path+"/"+inputName,path+"/"+outputName,path+"/"+filterName,str(intervall)])
    return res

path="./"
prime=2000000001 
K=5
N=12 
mapName="charToKMap.txt"
inputName="../texts/victorian/Eliot_Middlemarch.txt"
#inputName="../texts/modernism/Joyce_Ulysses.txt"
outputName="output.txt"
filterName="filter.txt"
intervall=10000
print NoveltyFilter(path,prime,K,N,mapName,inputName,outputName,filterName,intervall)

from pandas import *
path = "./"
filename="output.txt"
data=read_csv("{0}/{1}".format(path,filename), sep=",",na_values=[""," "],header=None,prefix="X")
plot(data["X2"])

path="./"
prime=2000000001 
K=5
N=12 
mapName="charToKMap.txt"
inputName="input.txt"
outputName="outputSecondRound.txt"
filterName="filter.txt"
intervall=10000
print NoveltyFilter(path,prime,K,N,mapName,inputName,outputName,filterName,intervall)

filename="outputSecondRound.txt"
data=read_csv("{0}/{1}".format(path,filename), sep=",",na_values=[""," "],header=None,prefix="X")
plot(data["X2"])

