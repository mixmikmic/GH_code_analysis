import pandas as pd
import datetime as dt
import numpy as np
import statsmodels.api as sm
import sklearn as sk
import sklearn.linear_model as lm

import sklearn.model_selection as split
from sklearn.pipeline import Pipeline

# Styling purpose
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("talk")

import Credit_risk_classes as Classes

from importlib import reload
reload(Classes)

# Processing will have to be in python
data_raw=pd.read_csv("/Users/jeroenderyck/Documents/Data/CreditRiskModels/Credit_DATA.csv",index_col=0)

# --> clean loans with only one observation
#DF is dataframe of raw dataset
data_raw=data_raw[data_raw.groupby("masterloanidtrepp")["observation_date"].transform("count")>1]

#Imputations
data = (data_raw
    .rename(columns={
        "bad_flag_final_v3":"Target_Response",
        'appvalue': 'pure_appraisal',
        'fmrappvalue': 'total_value_property',
        'gr_appvalue': 'pure_appraisal_Growth',
        'obal': 'outstanding_scheduled_balance',
        'origloanbal': 'original_loan_balance',
        'mrfytdocc': 'percentage_occupied_rentspace',
        'oltv': 'origination_loan_to_value',
        'oterm': 'original_maturity',
        'priorfyncf': 'most_recent_ncf',
        'priorfydscrncf': 'recent_ncf_ratio_debtservice',
        'priorfynoi': 'recent_noi',
        'priorfyocc': 'recent_fiscal_occupied_rentspace',
        'priorfydscr': 'most_recent_fiscal_debt_service',
        'cltv_1': 'current_loan_to_value_indexed'
    })
    .drop(["appvalue_prior","mrappvalue","gr_mrappvalue","sample","changeinvalue","balact","gr_balact"
        ,"msa"], axis=1)
)


Classes.rank_nan(data)

Data_Splitter=Classes.Data_Splitter(data,Target="Target_Response",
                                    fraction_training=0.8,Group_variable="masterloanidtrepp")

Training_X,Training_Y= Data_Splitter.X_Training, Data_Splitter.Y_Training
Testing_X, Testing_Y = Data_Splitter.X_Testing ,Data_Splitter.Y_Testing

(data[
    ["units","rentarea"]]
    .drop_duplicates().dropna()
    .query(" units <= 200")
    .query("1000 <= rentarea <= 200000")
    .plot(kind="scatter",x="units",y="rentarea")
)

data[["units"]].isnull().sum()

#Check results#Impute Units
data =(Classes.Imputer(data)
            .impute_linear_model(target_column="units",
                                 training_frame=Data_Splitter.X_Training,
                                 independent_variable="rentarea",
                                 intercept=False,
                                 )
      )
# Impute Rentarea
data =(Classes.Imputer(data)
            .impute_linear_model(target_column  = "rentarea",
                                 training_frame = Data_Splitter.X_Training,
                                 independent_variable = "units",
                                 intercept=False
                                 )
      )

#Check results
Classes.rank_nan(data).head(10)

Classes.impute_med

# Fill original  loan balance with median of group loan
#fill pure appraisal growth with mean of group loan
data=Classes.impute_median(data,"original_loan_balance",Group_variable="masterloanidtrepp")
data=Classes.impute_mean(data,"pure_appraisal_Growth",Group_variable="masterloanidtrepp")
data= Classes.impute_mean(data,"recent_ncf_ratio_debtservice",Group_variable="masterloanidtrepp")
# Changes Pure and total Appraisal 
data=data.assign(Change__pure_appraisal=lambda x: x.groupby("masterloanidtrepp")["pure_appraisal"].pct_change())
data=data.assign(Change_total_appraisal=lambda x: x.groupby("masterloanidtrepp")["total_value_property"].pct_change())

#Check if origination loan to value makes sense with original appraisal value
data_mortgage_values=data.groupby("masterloanidtrepp")["pure_appraisal"].first()*data.groupby("masterloanidtrepp"
                                                            )["origination_loan_to_value"].first()/100

data_mortgage = pd.DataFrame(data_mortgage_values,columns=["original_mortgage_value"])

# differences are not that big we add  a column to our dataframe because this one has no NAN
if "original_mortgage_value" in data.columns:
    pass
else:
    data= data.join(data_mortgage[["original_mortgage_value"]],on="masterloanidtrepp",how="left")


Median_original_mortgage =  data["original_mortgage_value"].median()
data=data.assign(original_mortgage_value=data["original_mortgage_value"].fillna(Median_original_mortgage).values)
    

Classes.rank_nan(data).head(10)

data_relationship_plot=(data[["units","rentarea"]]
                       .query("units <= 3000"))
                       


sns.set()
sns.pairplot(data_relationship_plot)

from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

cols_to_transform = [ 'maturitytype', 'rateindex', 'segment_2', 'segment_1_new','division','interestonly']

data_for_imputation= pd.get_dummies(data=data,columns=cols_to_transform).drop("observation_date",axis=1)

data_filled_soft =SoftImpute().complete(data_for_imputation)

data_imputed = pd.DataFrame(data=data_filled_soft,columns=data_for_imputation.columns,index=data_for_imputation.index)

data_imputed.to_csv("/Users/jeroenderyck/Desktop/Credit_Risk_Imputed")

data_relationship_plot=(data[["units","rentarea","rateindex"]]
                       .query("units <= 3000"))
                       


sns.set()
sns.pairplot(data_relationship_plot,hue="rateindex")

original_mortgage_value

from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

# X is the complete data matrix
# X_incomplete has the same values as X except a subset have been replace with NaN

# Use 3 nearest rows which have a feature to fill in each row's missing features
data_filled_knn = KNN(k=3).complete(data)

# matrix completion using convex optimization to find low-rank solution
# that still matches observed values. Slow!
data_filled_nnm = NuclearNormMinimization().complete(data)

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
data_filled_softimpute = SoftImpute().complete(data.normalize())




splitter= Classes.Data_Splitter(Data=data_imputed,fraction_training=0.8,Group_variable="masterloanidtrepp")

Training , Test= splitter.Training_frame, splitter.Validation_frame

Training.head()

Logistic = lm.LogisticRegression().fit(X=Training.drop("Target_Response",axis=1),y=Training["Target_Response"])

Logistic = Classes.LogisticRegressionModel().fit(Training,Response)

import sklearn.linear_model as lm

Test.colu

Logistic.predict(Test.drop("Target_Response",axis=1)).mean()

lm.LogisticRegression().fit

Test_Target = Test[["Target_Response"]]
Test_Target.mean()

Test[["Target_Response"]].mean()



