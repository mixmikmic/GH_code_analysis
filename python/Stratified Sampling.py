file = open('./ml-1m/users.dat', 'rb')
user_stratification = {
    'm_m': [],
    'm_a': [],
    'm_o': [],
    'f_m': [],
    'f_a': [],
    'f_o': []
}
cnt = 0
for line in file:
    cnt += 1
    #print(str(line).split("::")[0][2:])
    row_list = str(line).split("::")
    uid = int(row_list[0][2:])
    gender = row_list[1]
    age = int(row_list[2])
    
    if gender == 'M':
        if age == 1:
            user_stratification['m_m'].append(uid)
        elif age <= 35:
            user_stratification['m_a'].append(uid)
        elif age >= 45:
            user_stratification['m_o'].append(uid)
    elif gender == 'F':
        if age == 1:
            user_stratification['f_m'].append(uid)
        elif age <= 35:
            user_stratification['f_a'].append(uid)
        elif age >= 45:
            user_stratification['f_o'].append(uid)
print(cnt)
length = len(user_stratification['m_m'])             + len(user_stratification['m_a'])             + len(user_stratification['m_o'])             + len(user_stratification['f_m'])             + len(user_stratification['f_a'])             + len(user_stratification['f_o'])
print(length)

user_sampling = {
    '1': [],
    '2': [],
    '3': []
}
mm_bound1 = round(len(user_stratification['m_m'])/3)
mm_bound2 = round(2*len(user_stratification['m_m'])/3)

ma_bound1 = round(len(user_stratification['m_a'])/3)
ma_bound2 = round(2*len(user_stratification['m_a'])/3)

mo_bound1 = round(len(user_stratification['m_o'])/3)
mo_bound2 = round(2*len(user_stratification['m_o'])/3)

fm_bound1 = round(len(user_stratification['f_m'])/3)
fm_bound2 = round(2*len(user_stratification['f_m'])/3)

fa_bound1 = round(len(user_stratification['f_a'])/3)
fa_bound2 = round(2*len(user_stratification['f_a'])/3)

fo_bound1 = round(len(user_stratification['f_o'])/3)
fo_bound2 = round(2*len(user_stratification['f_o'])/3)

user_sampling['1'] = user_stratification['m_m'][0:mm_bound1] + user_stratification['m_a'][0:ma_bound1]                     + user_stratification['m_o'][0:mo_bound1] + user_stratification['f_m'][0:fm_bound1]                     + user_stratification['f_a'][0:fa_bound1] + user_stratification['f_o'][0:fo_bound1]

user_sampling['2'] = user_stratification['m_m'][mm_bound1:mm_bound2] + user_stratification['m_a'][ma_bound1:ma_bound2]                     + user_stratification['m_o'][mo_bound1:mo_bound2] + user_stratification['f_m'][fm_bound1:fm_bound2]                     + user_stratification['f_a'][fa_bound1:fa_bound2] + user_stratification['f_o'][fo_bound1:fo_bound2]
        
user_sampling['3'] = user_stratification['m_m'][mm_bound2:] + user_stratification['m_a'][ma_bound2:]                     + user_stratification['m_o'][mo_bound2:] + user_stratification['f_m'][fm_bound2:]                     + user_stratification['f_a'][fa_bound2:] + user_stratification['f_o'][fo_bound2:]

len(user_sampling['1'])+len(user_sampling['2'])+len(user_sampling['3'])

import scipy.io as sio
import numpy as np
sio.savemat('./User Stratification/users1.mat', {'users': np.array(user_sampling['1'])})
sio.savemat('./User Stratification/users2.mat', {'users': np.array(user_sampling['2'])})
sio.savemat('./User Stratification/users3.mat', {'users': np.array(user_sampling['3'])})
print('Done!')



