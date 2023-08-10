import pandas as pd
import numpy as np
headers = [
    'code','name','postcode','status_code','ccg','setting']
jun_2016 = pd.read_csv("2016_06_epraccur.csv", names=headers, usecols=[0,1,9,12,23,25]).set_index('code') 
mar_2017 = pd.read_csv("2017_03_epraccur.csv", names=headers, usecols=[0,1,9,12,23,25]).set_index('code') 

joined = jun_2016.join(mar_2017, how='inner',  rsuffix='2017')
joined.head()


changed_from_closed = joined[(joined.status_code != joined.status_code2017) & (joined.status_code == 'C')]
print "%s practices went from closed to something else" % changed_from_closed.status_code.count()
#changed[(changed.setting != changed.setting2017) & (changed.status_code2017 != 'C') & ((changed.setting == 4)|(changed.setting2017 == 4))]

print "             total dormant in Jun 2016: %6s" % len(joined[joined.status_code == 'D'])
print "      of which were closed by Mar 2017: %6s" % len(joined[(joined.status_code == 'D') & (joined.status_code2017 == 'C')])
print "of which had become active by Mar 2017: %6s" % len(joined[(joined.status_code == 'D') & (joined.status_code2017 == 'A')])
print "    of which still dormant at Mar 2017: %6s" % len(joined[(joined.status_code == 'D') & (joined.status_code2017 == 'D')]) 

joined[(joined.status_code == 'D') & (joined.status_code2017 == 'A')]

sept_2016 = pd.read_csv("2016_09_epraccur.csv", names=headers, usecols=[0,1,9,12,23,25]).set_index('code')
newly_dormant = sept_2016[(sept_2016.status_code == 'D') & (jun_2016.status_code != 'D')]

dormant_in_2016_09_codes = tuple(newly_dormant.index)
sql = """SELECT
  month,
  practice,
  SUM(items) AS items
FROM
  `hscic.prescribing`
WHERE
  month >= TIMESTAMP('2016-09-01') AND practice IN %s
GROUP BY
  month,
  practice
ORDER BY
  month, practice""" % str(dormant_in_2016_09_codes)

prescribing = pd.read_gbq(sql, project_id='ebmdatalab', dialect='standard')

prescribing[prescribing.month == '2016-09-01'].describe()

tmp  =[]
for practice, df in prescribing.groupby('practice'):
    tmp.append({'practice': practice, 'period': (df.month.max() - df.month.min()).days})
pd.DataFrame(tmp).describe()

pd.DataFrame(tmp).head()

# Plot some of them
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
for i, j in prescribing[prescribing.practice.isin(prescribing[prescribing.month > '2017-02-01'].practice)].groupby('practice'):
    j.set_index('month').plot(title=i)
    plt.show()

        
        
        
        
        

# Which still-dormant practices have had their CCG membership updated?
remained_dormant = joined[(joined.status_code == joined.status_code2017) & (joined.status_code == 'D') & (joined.ccg != joined.ccg2017)]
print len(remained_dormant)

# As a check - which have had CCG changes?
joined[(joined.ccg != joined.ccg2017)]
# Note: only one is a standard GP (setting = 4)

