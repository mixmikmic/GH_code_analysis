import pandas as pd

dataset=pd.read_csv("mashineLearningExport.csv")

dataset.head(20)

dataset.info()

dataset[dataset['BIRTHDAY'] == '1978-09-07']

dataset[dataset['DIPLOMPERCENTAGE'] == 58.0]

sortedData=dataset.sort(['IDENTIFICATION'])

sortedData

dataset[dataset['IDENTIFICATION']==6692]

dataset.get('IDENTIFICATION')

dataset.groupby('IDENTIFICATION')

dataset.groupby('IDENTIFICATION').sum()

dataset

dataset.groupby(['IDENTIFICATION']).count()

groupedData=dataset.groupby('IDENTIFICATION',sort=True).count()

groupedData

groupedData

groupedData.sort(columns='BIRTHDAY',ascending=False)

groupedData.sort_values(by='PROM',ascending=False)

groupedData=dataset.groupby(['IDENTIFICATION'],sort=True)

type(groupedData.size())

len(groupedData)

groupedData

sizeOfIndex=groupedData.size()

sizeOfIndex.sort_values(ascending=False)

listOfIndex=sizeOfIndex.tolist()

listOfIndex.count(1)

type(listOfIndex)

listOfIndex

IndexValueDf=pd.DataFrame(sizeOfIndex,columns=['ValueCount']) #this will get the dataframe with index and their valus count

IndexValueDf

IndexData1y= IndexValueDf[IndexValueDf['ValueCount']==1]#index of data with records for 1 year

len(IndexData1y)

IndexData2y= IndexValueDf[IndexValueDf['ValueCount']==2]#index of data with records for 2 year

len(IndexData2y)

IndexData3y= IndexValueDf[IndexValueDf['ValueCount']==3]#index of data with records for 3 year

len(IndexData3y)

IndexData4y= IndexValueDf[IndexValueDf['ValueCount']==4]#index of data with records for 4 year

len(IndexData4y)

IndexData4y

IndexData4y=IndexData4y.index.tolist()

type(IndexData4y.index)

IndexData4y

correctData4Y=dataset[dataset['IDENTIFICATION'].isin(IndexData4y)] #this dataset contain data from student with 4 years cursus

correctData4Y

def groupData(dataset):
    """this functuion will group my dataset into 4 categories according to index values and data per identification"""
    sizeOfIndex=dataset.groupby(['IDENTIFICATION'],sort=True).size() # we will return number of row for each values in the ideitification columsn
    IndexValueDf=pd.DataFrame(sizeOfIndex,columns=['ValueCount']) #this will get the dataframe with index and their valus count
    IndexData1y= IndexValueDf[IndexValueDf['ValueCount']==1].index.tolist()#index of data with records for 1 year
    IndexData2y= IndexValueDf[IndexValueDf['ValueCount']==2].index.tolist()
    IndexData3y= IndexValueDf[IndexValueDf['ValueCount']==3].index.tolist()
    IndexData4y= IndexValueDf[IndexValueDf['ValueCount']==4].index.tolist()
    #this dataset contain data from each student with one year cursus,2 year, 3 year,4 year
    return dataset[dataset['IDENTIFICATION'].isin(IndexData1y)],dataset[dataset['IDENTIFICATION'].isin(IndexData2y)], dataset[dataset['IDENTIFICATION'].isin(IndexData3y)],dataset[dataset['IDENTIFICATION'].isin(IndexData4y)]

Dataset1y,Dataset2y,Dataset3y,Dataset4y=groupData(dataset)

Dataset1y

Dataset2y

DataForMedecineFac=Dataset2y[Dataset2y.FAC=='Faculté de Médecine'] # this will save data only for medecine fac

len(DataForMedecineFac)

DataForMedecineFac

DataForMedecineFac1=Dataset1y[Dataset1y.FAC=='Faculté de Médecine'] #Faculté de Medecine avec une année

DataForMedecineFac1

DataForMedecineFac3=Dataset3y[Dataset3y.FAC=='Faculté de Médecine']

DataForMedecineFac3

Dataset3y

len(Dataset3y)

DataForMedecineFac1[DataForMedecineFac1.ACADYEAR=='2014-2015']

DataMedecine1YSat=DataForMedecineFac1.loc[(DataForMedecineFac1['ACADYEAR']=='2014-2015') & (DataForMedecineFac1['PROM']=='G1' )& ( (DataForMedecineFac1['MENT2']=='SATISFACTION' ) |( DataForMedecineFac1['MENT2']=='DISTINCTION' ))]

DataMedecine1YAjour15=DataForMedecineFac1.loc[(DataForMedecineFac1['ACADYEAR']=='2014-2015') & (DataForMedecineFac1['PROM']=='G1' )& (DataForMedecineFac1['MENT2']!='SATISFACTION' )]



DataMedecine1YSat16=DataForMedecineFac1[DataForMedecineFac1.ACADYEAR=='2015-2016' ]

DataMedecine1YSat16[ DataMedecine1YSat16.MENT2=='SATISFACTION'] # 

DataForMedecineFac1.shape

DataForMedecineFac.shape

dataset[dataset.FAC=='Faculté de Médecine'].shape





DataForMedecineFac=DataForMedecineFac.append( DataForMedecineFac1)

DataForMedecineFac.to_csv(path_or_buf='FacMedecineData.csv',na_rep='NaN')

DataMedecine1YAjour15=DataForMedecineFac1.loc[(DataForMedecineFac1['ACADYEAR']=='2014-2015') & (DataForMedecineFac1['PROM']=='G1' )& (DataForMedecineFac1['MENT2']!='SATISFACTION' )]

listProm=('G1','G0','G2','G3')



Dataset3y[Dataset3y['PROM'].isin([listProm])]

FacTech3y=Dataset3y.loc[Dataset3y.FAC=='Facult\xc3\xa9 des Sciences et Technologies Appliqu\xc3\xa9es']

FacTech3y

FacTech2y=Dataset2y.loc[Dataset2y.FAC=='Facult\xc3\xa9 des Sciences et Technologies Appliqu\xc3\xa9es']

FacTech2y

FacTech1y=Dataset1y.loc[Dataset1y.FAC=='Facult\xc3\xa9 des Sciences et Technologies Appliqu\xc3\xa9es']

FacTech1y



FacTech2y.loc[FacTech2y['PROM'].isin(['G0','G1'])]

FacTech3y.loc[FacTech3y['PROM'].isin(['G0','G1','G2'])].sort_values(by='IDENTIFICATION')

Dataset3y.loc[(Dataset3y['FAC']=='Faculté des Sciences et Technologies Appliquées') & Dataset3y['PROM'].isin(('G0','G1','G2'))]

Dataset3y.sort_values(by='IDENTIFICATION')

FacTech1y.sort_values(by='IDENTIFICATION')

FacTech2y.sort_values(by='IDENTIFICATION')

FacTech3y.sort_values(by='IDENTIFICATION')

FacTechnoData=FacTech1y.append(FacTech2y).append(FacTech3y)

FacTechnoData.shape

FacTechnoData.to_csv(path_or_buf='FacTechnoData.csv',na_rep='NaN')

type(listfac)

listfac=list(listfac)

listfac



FacPsy3y=Dataset3y.loc[Dataset3y['FAC']=="Facult\xc3\xa9 de Psychologie et des Sciences de l'\xc3\x89ducation"]

FacPsy3y

FacPsy2y=Dataset2y.loc[Dataset2y['FAC']=="Facult\xc3\xa9 de Psychologie et des Sciences de l'\xc3\x89ducation"]

FacPsy2y

FacPsy1y=Dataset1y.loc[Dataset1y['FAC']=="Facult\xc3\xa9 de Psychologie et des Sciences de l'\xc3\x89ducation"]

FacPsy1y

FacPsyCoData=dataset.loc[dataset['FAC']=="Facult\xc3\xa9 de Psychologie et des Sciences de l'\xc3\x89ducation"]

FacPsyCoData.to_csv(path_or_buf='FacPsycoData.csv',na_rep='NaN')

FacEcoData1year=Dataset1y.loc[Dataset1y['FAC']=='Facult\xc3\xa9 des Sciences \xc3\x89conomiques et de Gestion']

FacEcoData1year.shape

set(FacEcoData1year['ACADYEAR'].tolist())

FacEcoData2year=Dataset2y.loc[Dataset2y['FAC']=='Facult\xc3\xa9 des Sciences \xc3\x89conomiques et de Gestion']

FacEcoData2year.shape

set(FacEcoData2year['ACADYEAR'].tolist())

FacEcoData3year=Dataset3y.loc[Dataset3y['FAC']=='Facult\xc3\xa9 des Sciences \xc3\x89conomiques et de Gestion']

FacEcoData3year.shape

set(FacEcoData3year['ACADYEAR'].tolist())

FacEcoData4year=Dataset4y.loc[Dataset4y['FAC']=='Facult\xc3\xa9 des Sciences \xc3\x89conomiques et de Gestion']

FacEcoData4year.shape

FacEcoData1year.to_csv(path_or_buf='FacEcoData1year.csv',na_rep='NaN')
FacEcoData2year.to_csv(path_or_buf='FacEcoData2year.csv',na_rep='NaN')
FacEcoData3year.to_csv(path_or_buf='FacEcoData3year.csv',na_rep='NaN')

FacEcoData4year.to_csv(path_or_buf='FacEcoData4year.csv',na_rep='NaN')

#That all for toda



set(list(dataset.get('MENT1')))

set(list(dataset.get('MENT2')))

get_ipython().magic('matplotlib inline')

dataset.get('MENT1').hist()



