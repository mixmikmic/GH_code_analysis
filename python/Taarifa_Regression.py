import pandas as pd # for data import and dissection
import numpy as np # for data analysis
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

plt.interactive(False)
sns.set(style="whitegrid",color_codes=True)

# Reading the data where low_memory=False increases the program efficiency
data= pd.read_csv("data-taarifa.csv", low_memory=False)
sub1=data.copy()

list(sub1.columns.values)

#lowercase all variables
sub1.columns = [x.lower() for x in sub1.columns]

sub1.head(5)

## To fill every column with its own most frequent value you can use
sub1 = sub1.apply(lambda x:x.fillna(x.value_counts().index[0]))

from sklearn import preprocessing
le_enc = preprocessing.LabelEncoder()
#to convert into numbers
sub1.permit = le_enc.fit_transform(sub1.permit)
sub1.extraction_type_class=le_enc.fit_transform(sub1.extraction_type_class)
sub1.payment_type=le_enc.fit_transform(sub1.payment_type)
sub1.quality_group=le_enc.fit_transform(sub1.quality_group)
sub1.quantity_group=le_enc.fit_transform(sub1.quantity_group)
sub1.waterpoint_type_group=le_enc.fit_transform(sub1.waterpoint_type_group)
sub1.water_quality=le_enc.fit_transform(sub1.water_quality)
sub1.source_class=le_enc.fit_transform(sub1.source_class)
sub1.status_group=le_enc.fit_transform(sub1.status_group)

sub1.head(5)

print ("OLS regresssion model for the association between water pump condition status and quality of water in it")
reg1=smf.ols('status_group~permit',data=sub1).fit()
print (reg1.summary())

# Now, I continue to add the variables to this model to check for any loss of significance
print ("OLS regresssion model for the association between status_group and other variables of interest")
reg1=smf.ols('status_group~quantity_group+extraction_type_class+waterpoint_type_group',data=sub1).fit()
print (reg1.summary())

print ("OLS regresssion model for the association between water pump status group and all variables of interest")
reg1=smf.ols('status_group~extraction_type_class+payment_type+quality_group+quantity_group+waterpoint_type_group+source_class+permit+water_quality',data=sub1).fit()
print (reg1.summary())

scat1 = sns.regplot(x="status_group", y="quality_group", order=2, scatter=True, data=sub1)
plt.xlabel('Water pump status')
plt.ylabel ('quality of the water')
plt.title ('Scatterplot for the association between water pump status and water quality')
#print scat1
plt.show()



