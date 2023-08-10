instance_types  = ['c3.xlarge', 'c3.2xlarge', 'c3.4xlarge', 'c3.8xlarge']
region = 'us-east-1'
number_of_days = 10

end = get_ipython().getoutput('date -u "+%Y-%m-%dT%H:%M:%S"')
end = end[0]
start = get_ipython().getoutput('date -v "-{number_of_days}d" -u "+%Y-%m-%dT%H:%M:%S"')
start = start[0]
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

plt.figure(1, figsize(20,5))
for azName, azData in df[df.InstanceType=='c3.8xlarge'].groupby(['AvailabilityZone'], as_index=False):
    plt.plot(azData.index, azData['SpotPrice'],label=azName)
plt.legend()
plt.title('Spot Pricing - c3.8xlarge')
plt.show()

plt.figure(1, figsize(20,5))
for inName, inData in df[df.AvailabilityZone=='us-east-1a'].groupby(['InstanceType'], as_index=False):
    plt.hist(inData['SpotPrice'], bins=1000,label=inName)
    plt.xlim([0,1])
plt.legend()
plt.title('Histogram of Spot Pricing - us-east-1a')
plt.show()

df.groupby(['InstanceType'], as_index=False).agg([mean, std, min, max])



eight = df[df.InstanceType=='c3.xlarge']
eight.groupby(eight.index.hour).agg([mean, std]).plot(title='average price by hour of day (UTC) for c3.xlarge')

eight = df[df.InstanceType=='c3.xlarge']
eight.groupby(eight.index.dayofweek).agg([mean, std]).plot(title='average price by day of week (UTC) for c3.xlarge')



