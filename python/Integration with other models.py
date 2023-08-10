get_ipython().magic('matplotlib inline')

import numpy as np
from car_model import energy_consumption
import matplotlib.pyplot as plt

efficiencies = np.linspace(0.8, 0.9, 1000)
results = np.array([energy_consumption(
    1300, efficiency, 2.41, 0.721, 80, 0.5, 10, 0.01, 0.3, 0.875
    ) for efficiency in efficiencies
])

plt.scatter(efficiencies, results)

variables = np.linspace(0, 50, 1000)
results = np.array([energy_consumption(
    1300, 0.85, 2.41, 0.721, 80, 0.9, v, 0.01, 0.3, 0.875
    ) for v in variables
])

plt.scatter(variables, results)

variables = np.linspace(1.5, 5, 1000)
results = np.array([energy_consumption(
    1300, 0.85, v, 0.721, 80, 0.5, 10, 0.01, 0.3, 0.875
    ) for v in variables
])

plt.scatter(variables, results)

import brightway2 as bw

bw.projects.set_current("bw2_seminar_2017")

cars = bw.ExcelImporter("data/cars.xlsx")
cars.apply_strategies()
cars.match_database("ecoinvent 2.2", fields=['name', 'unit', 'location'])
assert not cars.statistics()[-1]
cars.write_database()

driving = bw.get_activity(('Electric cars', 'driving'))

for ex in driving.technosphere():
    print(ex)

energy_consumption(1300, 0.85, 2.41, 0.721, 80, 0.5, 10, 0.01, 0.3, 0.875)

lca = bw.LCA(
    {('Electric cars', 'driving'): 1}, 
    method=('ReCiPe Endpoint (E,A)', 'human health', 'particulate matter formation')
)
lca.lci()

lca.activity_dict

row = lca.product_dict[('Electric cars', 'electricity')]
col = lca.activity_dict[('Electric cars', 'driving')]

our_row = []

for i, obj in enumerate(lca.tech_params):
    if obj['row'] == row and obj['col'] == col:
        our_row.append(i)
        
assert len(our_row) == 1
our_row = our_row[0]
our_row

lca.tech_params[our_row]['amount'] = energy_consumption(1300, 0.85, 2.41, 0.721, 80, 0.5, 10, 0.01, 0.3, 0.875)
lca.rebuild_technosphere_matrix(lca.tech_params['amount'])

lca.lci_calculation()
lca.lcia()
lca.score

lca = bw.LCA({('Electric cars', 'driving'): 1}, method=('ReCiPe Endpoint (E,A)', 'human health', 'particulate matter formation'))
lca.lci()
lca.lcia()
lca.score



