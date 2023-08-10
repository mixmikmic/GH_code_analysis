import pandas as pd

MedDataset=pd.read_csv("FacMedecineData.csv")

MedDataset.shape

MedDataset.head(10)

DataGrouped=MedDataset.groupby(['IDENTIFICATION'])

DataGrouped.groups.get(1670)

colonnes=list(MedDataset.columns)

InputCol=colonnes[1:15]

OutCol=colonnes[15:]

OutCol

InputCol

groupedINOU=MedDataset.groupby(InputCol)

for Id , Data in grouped:
    print Id
    print Data

groupById=MedDataset[:5].groupby(InputCol[0]) #data grouped by id

groupById

for Id , Data in groupById:
    print Id
    print Data

groupById.groups.get(10010)

for Id, Name in groupedINOU:
    print (Id)
    print (Name)

groupById.head(2)

groupedINOU=MedDataset.groupby(InputCol)

groupedINOU.count()

MedDataset.rename( columns={"SEC":"SEX"},inplace=True)

MedDataset

"""aggregateDip={
    for 
}#this will aggreagata data for diploms"""

InputCol

PersonalCol=InputCol[:3]
DiplomCol=InputCol[3:10]
SchoolCol=InputCol[10:]

PersonalCol

DiplomCol

SchoolCol

len(MedDataset.groupby(PersonalCol))

len(DataGrouped)

len(MedDataset.groupby(PersonalCol ))

PersonalCol + DiplomCol

MedDataset.head(20)

MedDataset.info()

ColWIthoutInfo=['DIPLOMDATE','DIPLOMMENTION','DIPLOMPLACE','SCHOOLCODE']

Medataset1=MedDataset.drop(labels=ColWIthoutInfo,axis=1,inplace=False) #the dataset with good columns means less missing values

Medataset1

Medataset1.drop(labels='Unnamed: 0',axis=1,inplace=True)

Medataset1['SCHOOLSTATUS'].value_co

Medataset1.rename(columns={'IDENTIFICATION':'ID'},inplace=True)

Medataset1

len(Medataset1.groupby('ID').count())

ColumnsName1=list(Medataset1.columns)

ColumnsName1

PersonalCol1=ColumnsName1[:3]
DiplomCol1=ColumnsName1[3:7]
SchoolCol1=ColumnsName1[7:10]

PersonalCol1

DiplomCol1

SchoolCol1

len(Medataset1.groupby(PersonalCol1 + DiplomCol1 + SchoolCol1))

InputColomns1=ColumnsName1[:10] #The input the data will be grouped 
OupColomns2=ColumnsName1[10:] #the output

InputColomns1

OupColomns2

GroupedData=Medataset1.groupby(InputColomns1) #we want to group data by InputColums 1

GroupedData.count()

agrregate={
    "ACADYEAR":{
        "START":"first",
        "END":"last",
        "NUMBER":'count'
    },
    "PERC1":{
    "1year":'first',
    "2ndYear":'last'
    }, 
    #we will need to write function that will handle for multiple years this only works for 2
    "MENT1":{
    "1year":'first',
    "2ndYear":'last'
    },
    "PERC2":{
    "1year":'first',
    "2ndYear":'last',
    },
    "MENT2":{
    "1year":'first',
    "2ndYear":'last',
    },
    "FAC":'first',
    "OPT":'first',
    "PROM":{
        "START":'first',
        'END':"last"
    },   
}
#try to perform the aggregation with the aggregation functio
MedatasetSMF=GroupedData.agg(agrregate)

MedatasetSMF #this dataset contain data for Medecine Faculty Group and aggregate...

MedatasetSMF.columns

columnsList=MedatasetSMF.columns

columnsList

list(columnsList)

listCOlumns=[]
for item in columnsList:
    listCOlumns.append(item[0] + '-'+ item[1])
    

listCOlumns

IndexColums=pd.Index(listCOlumns)

IndexColums

MedatasetSMF.columns=IndexColums

MedatasetSMF

MedatasetSMF.Index()

DataIndex=MedatasetSMF.index

type(DataIndex)

list(DataIndex)[0]

type(MedatasetSMF.reset_index)

MedatasetSMF

list(MedatasetSMF.index)

MedatasetSMF.reset_index(inplace=True)

MedatasetSMF

MedatasetSMF.info()

#MedatasetSMF.to_csv(path_or_buf='MedecineDataSemiFInished.csv',na_rep='NaN')

set(list(MedatasetSMF.get('MENT1-1year')))

set(list(MedatasetSMF.get('MENT2-1year')))

MedDataset

MedDataset1

Medataset1

#Medataset1.to_csv(path_or_buf='Medataset1.csv',na_rep='NaN') 



