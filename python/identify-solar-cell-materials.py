import pandas as pd
import numpy as np
from scipy.stats import rankdata

results = pd.read_csv('meredig_bandgap_predictions.prop')
print('Loaded %d predictions'%len(results))

results.head()

results.query('bandgap_predicted >= 0.9 and bandgap_predicted <= 1.7', inplace=True)
print('Found %d entries with desired band gap'%len(results))

results['bandgap_distance'] = np.abs(results['bandgap_predicted'] - 1.3)

results['stability_score'] = rankdata(results[['ML_stability_measured','heuristic_stability_measured']].apply(np.mean, axis=1))

print('Best entries by stability:')
results.sort_values('stability_score', ascending=True).head(5)[['Entry', 'ML_stability_measured', 'heuristic_stability_measured']]

results['bandgap_score'] = rankdata(results['bandgap_distance'])

print('Best entries by band gap:')
results.sort_values('bandgap_score', ascending=True).head(5)[['Entry','bandgap_predicted']]

results['combined_score'] = results[['bandgap_score','stability_score']].apply(np.mean, axis=1)

print('Best materials by both scores:')
results.sort_values('combined_score', ascending=True).head(10)[['Entry','bandgap_predicted']]



