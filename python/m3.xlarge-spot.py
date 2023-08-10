instance_types  = ['m3.xlarge']
region = 'us-east-1'
number_of_days = 7

end = get_ipython().getoutput('date -u "+%Y-%m-%dT%H:%M:%S"')
end = end[0]
end = '2016-01-07T04:34:13'
start = get_ipython().getoutput('date -v-{number_of_days}d -u "+%Y-%m-%dT%H:%M:%S"')
start = start[0]
start = '2016-01-01T04:34:13'
print "will process from " + start + " to " + end

import sys
import boto as boto
import boto.ec2 as ec2
import datetime, time
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')

ec2 = boto.ec2.connect_to_region(region)


#
# process the output and convert to a dataframe
#

l = []
for instance in instance_types:
    sys.stdout.write("*** processing " + instance + " ***\n")
    sys.stdout.flush()
    prices = ec2.get_spot_price_history(start_time=start, end_time=end, instance_type=instance)
    for price in prices:
        d = {'InstanceType': price.instance_type, 
             'AvailabilityZone': price.availability_zone, 
             'SpotPrice': price.price, 
             'Timestamp': price.timestamp}
        l.append(d)
    next = prices.next_token
    while (next != ''):
        sys.stdout.write(".")
        sys.stdout.flush()
        prices = ec2.get_spot_price_history(start_time=start, end_time=end, instance_type=instance,
                                            next_token=next )
        for price in prices:
            d = {'InstanceType': price.instance_type, 
                 'AvailabilityZone': price.availability_zone, 
                 'SpotPrice': price.price, 
                 'Timestamp': price.timestamp}
            l.append(d)
        next = prices.next_token
        
    sys.stdout.write("\n")

df = pd.DataFrame(l)
df = df.set_index(pd.to_datetime(df['Timestamp']))

df['SpotPrice'].describe()

for k, g in df.sort(ascending=True).groupby(['InstanceType'], as_index=False):
    plt.figure(1, figsize(20,5))
    for key, grp in g.groupby(['AvailabilityZone'], as_index=False):
        plt.hist(grp['SpotPrice'], bins=1000, label=key)
        plt.xlim([0, 1])
        #grp.groupby(grp.index.dayofweek).agg(['mean']).plot()
    plt.legend()
    plt.title('Histogram of Spot Pricing - ' + k)
    plt.show()

for k, g in df.sort(ascending=True).groupby(['InstanceType'], as_index=False):
    plt.figure(1, figsize(20,5))
    for key, grp in g.groupby(['AvailabilityZone'], as_index=False):
        plt.plot(grp.index, grp['SpotPrice'], label=key)
        #grp.groupby(grp.index.dayofweek).agg(['mean']).plot()
        
    plt.legend()
    plt.title('Spot Pricing - ' + k)
    plt.show()



