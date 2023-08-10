import bayeslite
from bayeslite.read_pandas import bayesdb_read_pandas_df
import bdbcontrib
from bdbcontrib import cursor_to_df as df
from bdbcontrib.recipes import quickstart
import pandas as pd
import numpy
import re
import matplotlib
from matplotlib import ft2font
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from bdbcontrib.recipes import quickstart
import crosscat
import crosscat.MultiprocessingEngine as ccme
import bayeslite.metamodels.crosscat
import os

import pandas as pd
import crosscat
from bdbcontrib.recipes import quickstart
import bayeslite
from bayeslite.read_pandas import bayesdb_read_pandas_df

DATA = pd.read_csv("Most+Recent+Cohorts+(All+Data+Elements).csv", delimiter=',', low_memory = False)
DATA = DATA.replace('PrivacySuppressed', '-1')
#Restricting the dataframe to 57 variables of interest
df =DATA.loc[:,['OPEID','PFTFAC','UGDS','CCUGPROF','CCBASIC','locale2','region','st_fips','ACTCM75','ACTCMMID','ACTCM25','SAT_AVG_ALL','SATWR75','SATWRMID','SATWR25','SATMT75','SATMTMID','SATMT25','SATVR75','SATVRMID','SATVR25','ADM_RATE_ALL','WOMENONLY','MENONLY','AANAPII','PBI','md_earn_wne_p10','mn_earn_wne_p10','md_faminc','faminc','TUITIONFEE_PROG','AVGFACSAL','PCTPELL','IND_DEBT_MDN','HI_INC_DEBT_MDN','MD_INC_DEBT_MDN','LO_INC_DEBT_MDN','WDRAW_DEBT_MDN','GRAD_DEBT_MDN','MD_INC_RPY_5YR_RT','NOLOAN_COMP_ORIG_YR2','LOAN_COMP_ORIG_YR2','NOPELL_COMP_ORIG_YR2','PELL_COMP_ORIG_YR2','HI_INC_COMP_ORIG_Y','MD_INC_COMP_ORIG_Y','LO_INC_COMP_ORIG_Y','DEATH_YR2_RT','NOLOAN_ENRL_ORIG_YR','NOLOAN_DEATH_YR2_RT','LOAN_DEATH_YR2_RT','NOPELL_DEATH_YR2_RT','PELL_DEATH_YR2_RT','HI_INC_DEATH_YR2_R','MD_INC_DEATH_YR2_R','LO_INC_DEATH_YR2_R','DEP_DEBT_MDN']] 
#Renaming the variables names by their definition
df.columns = ['8-digit OPE ID for institution','Faculty Rate','Enrollment of undergraduate degree-seeking students','Carnegie Classification -- undergraduate profile','Carnegie Classification -- basic','Degree of urbanization of institution','Region (IPEDS)','FIPS code for state','75th percentile of the ACT cumulative score','Midpoint of the ACT cumulative score','25th percentile of the ACT cumulative score','Average SAT equivalent score of students admitted for all campuses rolled up to the 6-digit OPE ID','75th percentile of SAT scores at the institution (writing)','Midpoint of SAT scores at the institution (writing)','25th percentile of SAT scores at the institution (writing)','75th percentile of SAT scores at the institution (math)','Midpoint of SAT scores at the institution (math)','25th percentile of SAT scores at the institution (math)','75th percentile of SAT scores at the institution (critical reading)','Midpoint of SAT scores at the institution (critical reading)','25th percentile of SAT scores at the institution (critical reading)','Admission rate for all campuses rolled up to the 6-digit OPE ID','Flag for women-only college','Flag for men-only college','Flag for Asian American Native American Pacific Islander-serving institution','Flag for predominantly black institution','Median earnings of students working and not enrolled 10 years after entry','Mean earnings of students working and not enrolled 10 years after entry','Median family income','Average family income','TUITIONFEE_PROG','Average faculty salary','Percentage of Pell Grant','The median debt for independent students','The median debt for students with family income between over 75k','The median debt for students with family income between $30k and 75k','The median debt for students with family income between $0 and 30k','The median debt for students who have not completed','The median debt for students who have completed','Five-year repayment rate by family income ($30k-75k)','Percent of students who never received a federal loan at the institution and who were still enrolled at original institution within 2 years','Percent of students who received a federal loan at the institution and who completed in 2 years ','Percent of students who did not receive a Pell Grant at the institution and who completed in 2 years at original ','Percent of students who received a Pell Grant at the institution and who completed in 2 years at original ','Percent of high-income (over in nominal family income) students who died within a year','Percent of middle-income (between $30k and 75k in nominal family income) students who died within a year','Percent of female students who transferred to a 2-year institution and whose status is unknown within 8 years','Percent died within 2 years at original institution','Percent of students who never received a federal loan at the institution and who were still enrolled at original institution within a year','Percent of students who never received a federal loan at the institution and who died within 2 years at original institution','Percent of students who received a federal loan at the institution and who died within 2 years at original institution','Percent of students who never received a Pell Grant at the institution and who died within 2 years at original institution','Percent of students who received a Pell Grant at the institution and who died within 2 years at original institution','Percent of high income (more than 75k in nominal family income) students who died within 2 years','Percent of middle-income (between $30k and 75k in nominal family income) students who died within 2 years','Percent of low income (between 0 and 30k in nominal family income) students who died within 2 years','The median debt for dependent students']
df = df.iloc[:,:]

DATA = pd.read_csv("Most+Recent+Cohorts+(All+Data+Elements).csv", delimiter=',', low_memory = False)
DATA = DATA.replace('PrivacySuppressed', '-1')
#Restricting the dataframe to 57 variables of interest
data =DATA.loc[:,['INSTNM','OPEID','UGDS','ADM_RATE_ALL','faminc','TUITIONFEE_PROG','DEP_DEBT_MDN']] 
data = data.set_index(['INSTNM'])
#Renaming the variables names by their definition
data.columns = ['8-digit OPE ID for institution','Enrollment of undergraduate degree-seeking students',
                'Admission rate for all campuses rolled up to the 6-digit OPE ID', 'Average family income',
                'TUITIONFEE_PROG','The median debt for dependent students']
data = data.iloc[:,:]

bdb = bayeslite.bayesdb_open("sim.bdb")

bdb = bayeslite.bayesdb_open("similarities.bdb")

bayesdb_read_pandas_df(bdb, "df", data, create=True)

# Link your dataframe to your bdb file
bayesdb_read_pandas_df(bdb, "simi", df, create=True)

# Load the education dataset into a local instance of bayeslite
ed = quickstart(name='df', bdb_path='sim.bdb')
q = ed.q

# Load the education dataset into a local instance of bayeslite
ed = quickstart(name='simi', bdb_path='similarities.bdb')
q = ed.q

ed.analysis_status()

ed.analyze(models = 4, iterations=10)

q('''
ESTIMATE SIMILARITY WITH RESPECT TO "Enrollment of undergraduate degree-seeking students" FROM PAIRWISE %g
''')

q('''
select * from df where ("8-digit OPE ID for institution" = "00215500") ''')

q('''
Estimate SIMILARITY WITH RESPECT TO (list of variable comma separated) FROM PAIRWISE ROWS OF df''')

q(''' 
ESTIMATE SIMILARITY TO ("8-digit OPE ID for institution" = "00217800") WITH RESPECT TO 
("Enrollment of undergraduate degree-seeking students",
                "Admission rate for all campuses rolled up to the 6-digit OPE ID", "Average family income",
                "TUITIONFEE_PROG","The median debt for dependent students")
FROM %g''')

q('''
ESTIMATE SIMILARITY TO ("8-digit OPE ID for institution" = "00217800")
WITH RESPECT TO
("Enrollment of undergraduate degree-seeking students",
                "Admission rate for all campuses rolled up to the 6-digit OPE ID", "Average family income",
                "TUITIONFEE_PROG","The median debt for dependent students")
FROM %g WHERE ("8-digit OPE ID for institution" = "00215500")''')

q('''
ESTIMATE SIMILARITY TO ("8-digit OPE ID for institution" = "00217800")
WITH RESPECT TO "TUITIONFEE_PROG"
FROM %g WHERE ("8-digit OPE ID for institution" = "00215500")''')

q('''
ESTIMATE SIMILARITY TO ("8-digit OPE ID for institution" = "00217800")
WITH RESPECT TO "Admission rate for all campuses rolled up to the 6-digit OPE ID"
FROM %g WHERE ("8-digit OPE ID for institution" = "00215500")''')

DATA = pd.read_csv("Most+Recent+Cohorts+(All+Data+Elements).csv", delimiter=',', low_memory = False)

DATA



sim_1_var = q('''
ESTIMATE "INSTNM", SIMILARITY WITH RESPECT TO ("TUITIONFEE_PROG") as value
FROM PAIRWISE %g
ORDER BY value DESC 
''')

sim_mit = q(''' 
ESTIMATE "INSTNM", SIMILARITY TO ("key" = "00217800") WITH RESPECT TO 
("Enrollment of undergraduate degree-seeking students",
                "Admission rate for all campuses rolled up to the 6-digit OPE ID", "Average family income",
                "TUITIONFEE_PROG","The median debt for dependent students") as value
FROM %g
ORDER BY value DESC ''')









