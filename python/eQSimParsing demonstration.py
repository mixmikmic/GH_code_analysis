get_ipython().magic('matplotlib inline')

get_ipython().magic('run eqsimparsing.py')

import seaborn as sns
sns.set(style="white", context="talk")

pv_a_dict.keys()

type(pv_a_dict['CHILLERS'])

pv_a_dict['PUMPS']

pv_a_dict['PUMPS'].ix[:, 'W/GPM'].plot(kind='bar', figsize=(16,9), title='Pump W/GPM', color='#EB969C',);
sns.despine()
plt.show();

pv_a_dict['BOILERS']

sv_a_dict['Fans'].ix[:,'W/CFM'].plot(kind='barh', figsize=(12,10), color='#EB969C', title='Fan W/CFM')
sns.despine()
plt.show();

sv_a_dict.keys()

sys = sv_a_dict['Systems']
sys.head()



# Calculating Max People/sqft
sys['Max People/sqft'] = sys['Max People'] / sys['Floor Area (sqft)']

# Sorting and returning top 10
sys.sort_values(by='Max People/sqft', ascending=False).head(10)

zones = sv_a_dict['Zones']
zones.head(10)

def custom_apply_zones(x):
    """ Aggregate zone data to the system level
    
    For the zones, some columns should be summed (CFM, Capacity, etc)
    But others should be averaged
    """
    # For these three columns, do a mean
    if x.name in ['Minimum Flow (Frac)', 'Sensible (FRAC)', 'W/CFM']:
        return np.mean(x)
    # For the rest, do a sum
    else:
        return np.sum(x)

# After the groupby, the apply applies to each group dataframe. So I use a lambda x to apply to each column
zones_agg_metrics = zones.groupby(level='System').apply(lambda x: x.apply(custom_apply_zones))
# Recalc a weighted W/CFM
zones_agg_metrics['W/CFM'] = zones_agg_metrics['Fan (kW)'] * 1000 / zones_agg_metrics['Supply Flow (CFM)']

zones_agg_metrics.head()

fans = sv_a_dict['Fans']

fans

fans.groupby(level='System').apply(sum)['Power Demand (kW)'].sort_values(inplace=False).plot(kind='barh',
                                                                figsize=(12,10), color='#EB969C',
                                                                title='System wide fan power demand (kW)')
sns.despine()
plt.show();

