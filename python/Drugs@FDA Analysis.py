get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib

df = pd.read_csv('masterDrugList2.csv')
cols = [ 'Name','appNo', 'appType','approvDate','Company','marketStat','reviewAvailable','reviewPageLink','medReviewAvailable','statReviewAvailable','sumReviewAvailable','PatientPopulationAltered','PPAReviewAvailable','PPAReviewLink']
df = df[cols]

df

df.Name.count()

df.groupby(df.marketStat).marketStat.count()

iDs = df[(df.marketStat != 'Discontinued') & (df.appType != 'ANDA') & (df.appType != 'None (Tentative Approval)')] #interest drugs
iDs.date = pd.to_datetime(iDs.approvDate[iDs.approvDate != '-'])
iDs.head()

iDs.Name.count()

propRevAv = float(iDs[iDs.reviewAvailable == 'True'].reviewAvailable.count())/float(iDs.reviewAvailable.count())
propPPAAv = float(iDs[(iDs.PatientPopulationAltered == 'True') & (iDs.PPAReviewAvailable == 'True')].reviewAvailable.count())/float(iDs[iDs.PatientPopulationAltered == 'True'].PatientPopulationAltered.count())
propMedRevAv = float(iDs[iDs.medReviewAvailable == 'True'].medReviewAvailable.count())/float(iDs.medReviewAvailable.count())
propStatRevAv = float(iDs[iDs.statReviewAvailable == 'True'].statReviewAvailable.count())/float(iDs.statReviewAvailable.count())
propSumRevAv = float(iDs[iDs.sumReviewAvailable == 'True'].sumReviewAvailable.count())/float(iDs.sumReviewAvailable.count())

summaryTable = pd.DataFrame([pd.Series(['Main Reviews','PPA Reviews','Medical Reviews','Statistical Reviews','Summary Reviews']),pd.Series([propRevAv,propPPAAv,propMedRevAv,propStatRevAv,propSumRevAv])]).T
summaryTable.columns = ['Review Type','Proportion Available']
summaryTable

compUnavRev = iDs[iDs.reviewAvailable != 'True'].reviewAvailable.groupby(iDs.Company).count()
compUnavRev.name = 'No. of unavailable reviews'
compAvRev = iDs[iDs.reviewAvailable == 'True'].reviewAvailable.groupby(iDs.Company).count()
compAvRev.name = 'No. of available reviews'
companies = pd.concat([compUnavRev,compAvRev],axis=1).sort_values(by='No. of unavailable reviews',ascending=False).fillna(0)
companies

count = iDs.date.groupby(iDs.date.dt.year).count()
count.plot(kind="bar",figsize=(20,10),grid=True, title='Number of approved applications by year', color='k')

iDs.propRevAv_yr = iDs[iDs.reviewAvailable == 'True'].reviewAvailable.groupby(iDs.date.dt.year).count()/iDs.reviewAvailable.groupby(iDs.date.dt.year).count()
iDs.propRevAv_yr = iDs.propRevAv_yr.fillna(0) #changes NaN values to 0
iDs.propRevAv_yr.plot(kind="bar",figsize=(20,10),grid=True, title='Proportion of Main Reviews available by year', color='g')

iDs.propMedRevAv_yr = iDs[iDs.medReviewAvailable == 'True'].medReviewAvailable.groupby(iDs.date.dt.year).count()/iDs.medReviewAvailable.groupby(iDs.date.dt.year).count()
iDs.propMedRevAv_yr = iDs.propMedRevAv_yr.fillna(0) #changes NaN values to 0
iDs.propMedRevAv_yr.plot(kind="bar",figsize=(20,10),grid=True, title='Proportion of Medical Reviews available by year', color='y')

iDs.propStatRevAv_yr = iDs[iDs.statReviewAvailable == 'True'].statReviewAvailable.groupby(iDs.date.dt.year).count()/iDs.statReviewAvailable.groupby(iDs.date.dt.year).count()
iDs.propStatRevAv_yr = iDs.propStatRevAv_yr.fillna(0) #changes NaN values to 0
iDs.propStatRevAv_yr.plot(kind="bar",figsize=(20,10),grid=True, title='Proportion of Statistical Reviews available by year',color='m')

iDs.propSumRevAv_yr = iDs[iDs.sumReviewAvailable == 'True'].sumReviewAvailable.groupby(iDs.date.dt.year).count()/iDs.sumReviewAvailable.groupby(iDs.date.dt.year).count()
iDs.propSumRevAv_yr = iDs.propSumRevAv_yr.fillna(0) #changes NaN values to 0
iDs.propSumRevAv_yr.plot(kind="bar",figsize=(20,10),grid=True, title='Proportion of Summary Reviews available by year')

iDs.propPPAAv_yr = iDs[(iDs.PatientPopulationAltered == 'True') & (iDs.PPAReviewAvailable == 'True')].PPAReviewAvailable.groupby(iDs.date.dt.year).count()/(iDs[iDs.PatientPopulationAltered == 'True'].PPAReviewAvailable.groupby(iDs.date.dt.year).count())
iDs.propPPAAv_yr = iDs.propPPAAv_yr.fillna(0) #changes NaN values to 0
iDs.propPPAAv_yr.plot(kind="bar",figsize=(20,10),grid=True, title='Proportion of PPA Reviews available by year',color='c')

