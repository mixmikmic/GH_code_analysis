get_ipython().magic('pylab inline')
import pysd
import numpy as np
import pandas as pd

motivation_model = pysd.read_vensim('../../models/Sales_Agents/Sales_Agent_Motivation_Dynamics.mdl')
market_model = pysd.read_vensim('../../models/Sales_Agents/Sales_Agent_Market_Building_Dynamics.mdl')

def runner(params):
    market = market_model.run(dict(params),return_columns=['Tenure'])
    motiv = motivation_model.run(dict(params),return_columns=['Tenure'])
    return pd.Series({'market':market['Tenure'].iloc[-1], 
                      'motivation':motiv['Tenure'].iloc[-1]})

base = runner({'Startup Subsidy': 0,
              'Startup Subsidy Length': 0})
base

subsidy = pd.DataFrame(np.arange(0,1,.05), columns=['Startup Subsidy'])
subsidy['Startup Subsidy Length'] = 3
subsidy.plot(subplots=True, kind='bar');

subsidy_res = subsidy.apply(runner, axis=1)/base

subsidy_res.index = subsidy['Startup Subsidy']
subsidy_res.plot(style='o-')
plt.ylabel('Improvement in Average Tenure over baseline')
plt.title('Changing the subsidy gives little discernment between theories');

l_subsidy = pd.DataFrame(np.arange(0,12,1), 
                       columns=['Startup Subsidy Length'])
l_subsidy['Startup Subsidy'] = .5
l_subsidy.plot(subplots=True);

l_subsidy = pd.DataFrame(index=range(20), 
                         data=0.5,
                         columns=['Startup Subsidy'])
l_subsidy['Startup Subsidy Length'] = range(20)
l_subsidy.plot(subplots=True, kind='bar')
plt.xlabel('Experiment Number');

l_subsidy_res = l_subsidy.apply(runner, axis=1)/base

l_subsidy_res.index = l_subsidy['Startup Subsidy Length']
l_subsidy_res.plot(style='o-')
plt.ylabel('Improvement in Average Tenure over baseline');
plt.title('Changing the subsidy length gives more discernment at longer subsidization');

total_subsidy = pd.DataFrame(np.arange(0.05,1,.05), 
                       columns=['Startup Subsidy'])
total_subsidy['Startup Subsidy Length'] = 10/total_subsidy['Startup Subsidy']
total_subsidy.plot(subplots=True, kind='bar');

total_subsidy_res = total_subsidy.apply(runner, axis=1)

total_subsidy_res.index = total_subsidy['Startup Subsidy']
total_subsidy_res.plot(style='o-')
plt.ylabel('Improvement in Average Tenure over baseline');

