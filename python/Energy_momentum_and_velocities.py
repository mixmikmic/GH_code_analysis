import math
#Question 3
#What is $\beta$ for an electron with momentum 1 keV, 1 MeV, 10 MeV, 100 MeV, 1000 MeV? 

#b=math.sqrt((.001/.511)**2+1)
#print b
x=[.001,1,10,100,1000]
for i in x:
    b=math.sqrt(1/((.511/i)**2+1))
    #b=math.sqrt((i/.511)**2+1)
    print "Beta for an electron at %f MeV is %f" %(i,b)


#Question 4
#Repeat the above for a muon, pion, kaon, and a proton

x=[.001,1,10,100,1000]
m=[.511,105.7,139.6,497.6,938.3]
electron=[]
muon=[]
pion=[]
kaon=[]
proton=[]
for j in m:
    if j<1:
        for i in x:
            e=math.sqrt(1/((j/i)**2+1))
            print "Beta for an electron at %f MeV is %f" %(i,e)
            electron.append(e)
    elif 100<j<130: 
        for i in x:
            m=math.sqrt(1/((j/i)**2+1))
            print "Beta for a muon at %f MeV is %f" %(i,m)
            muon.append(m)
    elif 130<j<300:
        for i in x:
            p=math.sqrt(1/((j/i)**2+1))
            print "Beta for a pion at %f MeV is %f" %(i,p)
            pion.append(p)
    elif 450<j<500:
        for i in x:
            k=math.sqrt(1/((j/i)**2+1))
            print "Beta for a kaon at %f MeV is %f" %(i,k)
            kaon.append(k)
    else:
        for i in x:
            Pr=math.sqrt(1/((j/i)**2+1))
            print "Beta for a proton at %f MeV is %f" %(i,Pr)
            proton.append(Pr)

#Question 5
#For the 5 particles above, make a plot of $\beta$ vs. momentum for the momentum range 0-10 GeV.

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

plt.plot(x,electron,'b-',markersize=6,label='Electron')
plt.plot(x,muon,'r-',markersize=6,label='Muon')
plt.plot(x,pion,'g-',markersize=6,label='Pion')
plt.plot(x,kaon,'k-',markersize=6,label='Kaon')
plt.plot(x,proton,'m-',markersize=6,label='Proton')

# Add labels with a bigger font than the default.
plt.xlabel('Momentum in MeV',fontsize=14)
plt.ylabel('Beta',fontsize=14)

# Change the plotting range for the xlimits (xlim) and ylimits (ylim).
plt.xlim(-3,1100)
plt.ylim(-1,1.5)

# Add a title
plt.title("beta vs. Momentum")

# Add a legend.
plt.legend()



