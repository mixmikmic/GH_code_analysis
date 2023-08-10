from EDdata import EDdata

import numpy as np
import pandas as pd

portsmouth_cols = {
'pas_id':'dept_patid',
'nhs_number':'hosp_patid',
'age':'age',
'gender':'gender',
'department':'site',
'date_of_attendance':'arrival_date',
'time_of_attendance':'arrival_time',
'mode_of_arrival':'arrival_mode',
'triage_time':'first_triage_time',
'dr1_seen':'first_dr_time',
'referred_to__first_referral':'first_adm_request_time',
'referred_to_at_point_of_discharge':'adm_referral_loc',
'departure_method':'departure_method',
'left_dept_time':'leaving_time',
'departure_method':'departure_method'
}

pd.read_csv('./../../3_Data/Patient Journey ED Data 22.01.2014 to 31.10.2015.csv')

pmED2 = EDdata('pmth2')

pmED2.importRAW('./../../3_Data/Patient Journey ED Data 22.01.2014 to 31.10.2015.csv',portsmouth_cols)

pmED2._dataRAW

pmED2._dataRAW = pmED2._dataRAW.iloc[0:100]

pmED2.create_datetime_columns()

np.str

pmED2._dataRAW



pmED = EDdata('pmth')

pmED.status()

pmED._dataRAW = pd.read_csv('../../3_Data/processed/pmthED.csv',parse_dates = ['arrival_time','leaving_time'])

x = pmED._dataRAW

x.head(1)

x.tail(1)

x.dtypes

pd.datetime(x['leaving_datetime'])

x['waiting_time'] = (x['leaving_datetime'] - x['arrival_datetime']) #/ pd.Timedelta('1 minute')

x.arrival_datetime[0].to_datetime()

pd.to_datetime(pd.Series(['2014-01-22 10:48:00']))



