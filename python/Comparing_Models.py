get_ipython().magic('pylab inline')
import pysd
import pandas as pd
import scipy.optimize

model = pysd.read_vensim('../../models/Emails/Emails_in_2_fixed_categories.mdl')
params = {'total_emails':1000,
          'easy_fraction':.5,
          'easy_reply_time':1,
          'hard_reply_time':10}

model.run(params,
          return_columns=['net_email_output', 'easy_reply', 'hard_reply']).plot()
plt.title('Timeseries output representing a distribution over time deltas')
plt.xlim(0,20);

response_times = pd.read_csv('../../data/Emails/days_to_join_conversation.csv', 
                             names=['id','Days'], index_col='id')
num_conversations = len(response_times)
print 'Loaded %i conversations'%num_conversations

counts, edges, _ = plt.hist(response_times['Days'], bins=range(25), 
                            histtype='stepfilled', alpha=.5)
data = pd.Series(data=counts, 
                 index=1.*(edges[:-1]+edges[1:])/2) #take the average location in the bin

data.plot(linewidth=2, color='k', xlim=(0,25), 
          title='Histogram of response times');
plt.xlabel('Days to first response');

param_names = ['easy_fraction', 'easy_reply_time', 'hard_reply_time']

def error(param_list, data, num_conversations):
    params = dict(zip(param_names, param_list))
    params['total_emails'] = num_conversations
    
    #run the model with the new parameters, returning the info we're interested in
    result = model.run(params=params,
                       return_columns=['net_email_output'],
                       return_timestamps=list(data.index.values))
    
    #the sum of the squared errors
    sse = sum((result['net_email_output'] - data)**2)
    return sse

error([.5, .5, 20], data, num_conversations)

params = {'easy_fraction':.8,
          'easy_reply_time':1.1,
          'hard_reply_time':7}

bounds = {'easy_fraction':[0.001,.999],
          'easy_reply_time':[0.001,21],
          'hard_reply_time':[.001,52]}

res = scipy.optimize.minimize(error, [params[key] for key in param_names],
                              args=(data, num_conversations),
                              method='L-BFGS-B',#'L-BFGS-B', #'SLSQP', #'TNC'
                              bounds=[bounds[key] for key in param_names])
res

params=dict(zip(param_names, res['x']))
params['total_emails'] = num_conversations

model.run(params,
          return_columns=['net_email_output', 'easy_reply', 'hard_reply']).plot()

data.plot(linewidth=2, color='k', xlim=(0,25), 
          title='Histogram of response times');
plt.xlabel('Days to first response');



