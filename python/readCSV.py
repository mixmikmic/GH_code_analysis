import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

def readRedatamCSV(asciiFile):
    f = open(asciiFile, 'r')
    areas = []
    measures = []
    for line in f:
        columns = line.strip().split()
        #print columns
        if len(columns) > 0:
            if 'RESUMEN' in columns[0] :
                break
            elif columns[0] == 'AREA':
                area = str.split(columns[2],',')[0]
                areas.append(area)
            elif columns[0] == 'Total':
                measure = str.split(columns[2],',')[2]
                if measure == '-':
                    measure = np.nan
                measures.append(measure)
    try:        
        data = pd.DataFrame({'area':areas,'measure':measures})
        return data
    except:
        print asciiFile

comunasFile = '/home/pipe/Dropbox/NYU/classes/Applied Data Science/adsProject/data/indecOnline/headEducYjobs/comuna.csv'
comunas = readRedatamCSV(comunasFile)

comunas.area

baseMadre = comunas.loc[comunas.measure==0,:]
ruta = '/home/pipe/Dropbox/NYU/classes/Applied Data Science/adsProject/data/indecOnline/MODELO1E/'
for i in comunas.area:
    archivoCSV = ruta + i + '.csv'
    data = readRedatamCSV(archivoCSV)
    baseMadre = baseMadre.append(data)

baseMadre.measure = baseMadre.measure.apply(float)

baseMadre[baseMadre.area=='020041801'] = np.nan

baseMadre.to_csv(ruta + 'modelo1e.csv',index=False)

baseMadre.dropna().measure.describe()



