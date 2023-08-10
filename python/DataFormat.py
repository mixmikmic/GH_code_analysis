# define a path to the directory where the data is stored
get_ipython().magic('cd PrecipitationData')

#import the required packages and activate the option to plot in between cells
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from calendar import monthrange
from datetime import date
get_ipython().magic('pylab inline')

# open the precipitation data files
Plu_Carania = open('PluviometriaTotalDiaria_CARANIA.txt').read().splitlines()
Plu_Huangascar = open('PluviometriaTotalDiaria_HUANGASCAR.txt').read().splitlines()
Plu_Nicolas = open('PluviometriaTotalDiaria_NICOLASFRANCO.txt').read().splitlines()
Plu_Pacaran = open('PluviometriaTotalDiaria_PACARAN.txt').read().splitlines()
Plu_Socsi = open('PluviometriaTotalDiaria_SOCSI.txt').read().splitlines()
Plu_Tanta = open('PluviometriaTotalDiaria_TANTA.txt').read().splitlines()
Plu_Vilca = open('PluviometriaTotalDiaria_VILCA.txt').read().splitlines()
Plu_Yauricocha = open('PluviometriaTotalDiaria_YAURICOCHA.txt').read().splitlines()
Plu_Yauyos = open('PluviometriaTotalDiaria_YAUYOS.txt').read().splitlines()

# we need to define a dictionary since months are in spanish and they have to be transformed to date
meses = {'enero     ': 1, 'febrero   ': 2, 'marzo     ': 3,'abril     ':4,
        'mayo      ': 5, 'junio     ': 6, 'julio     ': 7,'agosto    ':8,
        'septiembre': 9, 'octubre   ': 10, 'noviembre ': 11, 'diciembre ':12};

# this is the main function that deals with precipition data
def ordenardatosplu(f):
    a = []
    # first we create a list of lists 
    for l in range(len(f[1:])):
        a.append(f[l+1].split('\t'))
    # then we fill the empty spaces on the table
    for l in range(len(a)):
        if a[l][0] != '    ':
            yr = a[l][0]
        elif a[l][0].isspace():
            a[l][0] = yr
    # months are in spanish, we need to change them to numbers
    for l in range(len(a)):
        for m in meses:
            if a[l][1] == m:
                a[l][1] = meses[m]
    # we open a Pandas TimeSeries and store data for a given day
    ts = pd.Series()
    for l in range(len(a)):
        ndias = monthrange(int(a[l][0]), int(a[l][1]))[1]
        for d in range(ndias):
            if a[l][d+3].isspace():
                ts[date(int(a[l][0]), a[l][1], d+1)] = np.nan
            else:
                #xx = (a[l][d+3])
                ts[date(int(a[l][0]), int(a[l][1]), d+1)] = float(a[l][d+3])
    #from a Pandas TimeSeries we create a DataFrame
    d = pd.DataFrame(list(ts.values), index=ts.index, columns=["Ppt"])
    return d

#this function is not used. it deal with flow data
def ordenardatoshid(f):
    a = []
    # first we create a list of lists 
    for l in range(len(f[1:])):
        a.append(f[l+1].split('\t'))
    # then we fill the empty spaces on the table
    for l in range(len(a)):
        if a[l][0] != '':
            yr = a[l][0]           
        elif a[l][0] == '':
            a[l][0] = yr
        
    ts = pd.Series()
    for l in range(len(a)):
        ndias = monthrange(int(a[l][0]), int(a[l][1]))[1]
        for d in range(ndias):
            if a[l][d+3] == '':
                ts[date(int(a[l][0]), int(a[l][1]), d+1)] = np.nan
            else:
                ts[date(int(a[l][0]), int(a[l][1]), d+1)] = float(a[l][d+3])
    d = pd.DataFrame(list(ts.values), index=ts.index, columns=["Ppt"])
    return d

#this function is not used. it deal with precipitation on monthly basis
def ord_pte_uchu(f):
    a = []
    # hemos hecho una lista de listas con cada linea
    for l in range(len(f[1:])):
        a.append(f[l+1].split('\t'))
                
    ts = pd.Series()
    for l in range(len(a)):
        for d in range(12):
            if a[l][d+2] == '':
                ts[date(int(a[l][0]), d+1, 15)] = np.nan
            else:
                ts[date(int(a[l][0]), d+1, 15)] = float(a[l][d+2])
    d = pd.DataFrame(list(ts.values), index=ts.index, columns=["Ppt"])
    return d

#we apply the function for every file. I could do a for loop but I was lazy
Ord_Carania = ordenardatosplu(Plu_Carania)
Ord_Huangascar = ordenardatosplu(Plu_Huangascar)
Ord_Nicolas = ordenardatosplu(Plu_Nicolas)
Ord_Pacaran = ordenardatosplu(Plu_Pacaran)
Ord_Socsi = ordenardatosplu(Plu_Socsi)
Ord_Tanta = ordenardatosplu(Plu_Tanta)
Ord_Vilca = ordenardatosplu(Plu_Vilca)
Ord_Yauricocha = ordenardatosplu(Plu_Yauricocha)
Ord_Yauyos = ordenardatosplu(Plu_Yauyos)

#plot of the whole data for a determined station
Ord_Carania.plot()
xticks(rotation='vertical')

#We change the heading for name more related to the station since we want to merge the DataFrames
CanPpt_comp = Ord_Carania
CanPpt_comp = CanPpt_comp.rename(columns = {'Ppt':'Carania'})
CanPpt_comp.head()

#We add the data from the other DataFrames to the first Dataframe. Its quite easy when you know the structure
CanPpt_comp['Huangascar'] = Ord_Huangascar
CanPpt_comp['Nicolas'] = Ord_Nicolas
CanPpt_comp['Pacaran'] = Ord_Pacaran
CanPpt_comp['Socsi'] = Ord_Socsi
CanPpt_comp['Tanta'] = Ord_Tanta
CanPpt_comp['Vilca'] = Ord_Vilca
CanPpt_comp['Yauricocha'] = Ord_Yauricocha
CanPpt_comp['Yauyos'] = Ord_Yauyos

#We can see how is the final version of out table
CanPpt_comp.describe()

#We create a multiplot of the whole dataset. This plot is interesting because shows the missing data.
plt.close('all')
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1, sharex=True, sharey=True)
f.set_size_inches(18,12)

ax1.plot(CanPpt_comp['Carania'], color='#4B4C4E')
ax1.set_ylim(0,40)
ax1.set_xlim(0,16000)
ax1.set_title('Carania')

ax2.plot(CanPpt_comp['Huangascar'], color='#3F83B7')
ax2.set_title('Huangascar')

ax3.plot(CanPpt_comp['Nicolas'], color='#7198BC')
ax3.set_title('Nicolas')

ax4.plot(CanPpt_comp['Pacaran'], color='#7B7C7E')
ax4.set_title('Pacaran')

ax5.plot(CanPpt_comp['Socsi'], color='#19375E')
ax5.set_title('Socsi')

ax6.plot(CanPpt_comp['Tanta'], color='#19375E')
ax6.set_title('Tanta')

ax7.plot(CanPpt_comp['Vilca'], color='#19375E')
ax7.set_title('Vilca')

ax8.plot(CanPpt_comp['Yauricocha'], color='#19375E')
ax8.set_title('Yauricocha')

ax9.plot(CanPpt_comp['Yauyos'], color='#19375E')
ax9.set_title('Yauyos')

mensuales = CanPpt_comp.groupby(lambda m: m.month)
mensuales.mean()

#This is what we have to do to fill the missin data by the monthly average
CanPpt_comp2 = mensuales.fillna(mensuales.mean())

#We check that all columns have the same amount of values
CanPpt_comp2.describe()

#We plot again the data to see the filled values
plt.close('all')
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1, sharex=True, sharey=True)
f.set_size_inches(18,12)

ax1.plot(CanPpt_comp2['Carania'], color='#4B4C4E')
ax1.set_ylim(0,40)
ax1.set_xlim(0,16000)
ax1.set_title('Carania')

ax2.plot(CanPpt_comp2['Huangascar'], color='#3F83B7')
ax2.set_title('Huangascar')

ax3.plot(CanPpt_comp2['Nicolas'], color='#7198BC')
ax3.set_title('Nicolas')

ax4.plot(CanPpt_comp2['Pacaran'], color='#7B7C7E')
ax4.set_title('Pacaran')

ax5.plot(CanPpt_comp2['Socsi'], color='#19375E')
ax5.set_title('Socsi')

ax6.plot(CanPpt_comp2['Tanta'], color='#19375E')
ax6.set_title('Tanta')

ax7.plot(CanPpt_comp2['Vilca'], color='#19375E')
ax7.set_title('Vilca')

ax8.plot(CanPpt_comp2['Yauricocha'], color='#19375E')
ax8.set_title('Yauricocha')

ax9.plot(CanPpt_comp2['Yauyos'], color='#19375E')
ax9.set_title('Yauyos')



