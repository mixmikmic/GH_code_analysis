get_ipython().magic('pylab inline')

import seaborn as sns
sns.set_context('poster', font_scale=1.25)

sns.set_style('ticks')

get_ipython().magic('load_ext autoreload')

get_ipython().magic('autoreload 2')

import pandas as pd

import OD_growth_finder as od

experiment = od.OD_growth_experiment('./test_test.xlsx', s=0.2, constant_background=0.08)

growth_data = experiment.get_all_growth_rates(save_pictures=False)

growth_data = growth_data.loc[growth_data['max_index'] != 0, :]
growth_data = growth_data.loc[growth_data['growth_rate'] > 0.05, :]

sns.distplot(growth_data['doubling_time'])
sns.rugplot(growth_data['doubling_time'])

plt.savefig('phoebe_pic.png', dpi=300, bbox_inches='tight')

growth_data.to_excel('phoebe_rates.xls')

growth_data

growth_data.loc[growth_data['well'] == 'H2', :]['doubling_time']*60

growth_data['row'] = growth_data['well'].apply(lambda x: ord(x[0]) - 65)
growth_data['column'] = growth_data['well'].apply(lambda x: int(x[1:]))

growth_rate_image = np.zeros((growth_data['row'].max() + 1, growth_data['column'].max() + 1))

growth_rate_image[growth_data['row'], growth_data['column']] = growth_data['growth_rate']

min_growth_rate = np.min(growth_data['growth_rate'])
max_growth_rate = np.max(growth_data['growth_rate'])

plt.imshow(growth_rate_image, origin='upper', interpolation='None')
plt.colorbar()
plt.clim(min_growth_rate - 0.05, max_growth_rate)

plt.savefig('marco_growth_data.png', dpi=300, bbox_inches='tight')

experiment.plot_growth_prediction('O10')

experiment.plot_growth_prediction('O22')

