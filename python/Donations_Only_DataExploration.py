get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

donations = pd.DataFrame.from_csv('opendata_donations.csv', index_col=None).ix[:,0:23]
donations = donations.rename(columns=lambda x: x.strip()) # removing whitespaces from columns
donations = donations[(donations.donor_zip!='SC')&(donations.donor_zip!='NY')&(donations.donor_zip!='NJ')&(donations.donor_zip!='TX')]
#donations = donations[(donations.donor_city!='" everything\"')&(donations.donor_city!='" inventor\"')&(donations.donor_city!='" Flight Recorders. The other plotting sister is learning to play recorder\"')]
#donations = donations.dropna()[donations.donor_state != 'ozone park'] 
donations.head(5)

# Checking if Donation amounts form a normal distribution
mean = donations['donation_total'].mean()
variance = donations['donation_total'].var()
print 'Mean and Variance:',mean,variance
sigma = math.sqrt(variance)
x = np.linspace(-1000,1000)
plt.plot(x,mlab.normpdf(x,mean,sigma))
plt.show()

# 1) Plot between Donation State and Donation Amount
d = donations
# Removing states that have less than 100 donations
d = d[(d.donor_state != 'ozone park')&(d.donor_state != 'AA') & (d.donor_state != 'AS') & (d.donor_state != 'GU') & (d.donor_state !='MP') & (d.donor_state !='PR') & (d.donor_state != 'VI') & (d.donor_state != 'AP')]
d.loc[:,'donor_state']
d = d.groupby(by='donor_state')

d_mean = d['donation_total'].mean()
d_mean.plot(kind='bar',figsize=(20,5))

#2) Plot between IsTeacher and Donation Amount
d = donations.groupby(by='is_teacher_acct')
d_mean = d['donation_total'].mean()
d_mean.plot(kind='bar',figsize=(7,5))

d['donation_total'].count()

#3) Plot between payment type and Donation Amount
d = donations.groupby(by='payment_method')
d_mean = d['donation_total'].mean()
d_mean.plot(kind='bar',figsize=(20,5))

#4) Plot between Donation type and Donation amount
d = donations
d1 = d[d.for_honoree == 't']['donation_total']
d2 = d[d.via_giving_page == 't']['donation_total']
d3 = d[(d.for_honoree == 'f') & (d.via_giving_page == 'f')]['donation_total']
print d1.count(),d2.count(),d3.count()
print 'Average Donation amount if Honoree is included:',d1.mean()
print 'Average Donation amount via Giving page:',d2.mean()
print 'Average Donation amount for not in the category:',d3.mean()

