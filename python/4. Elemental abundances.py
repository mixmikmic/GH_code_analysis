import q2
data = q2.Data('hyades_solution_odfnew_err.csv', 'hyades_lines.csv')

star = q2.Star('hd29310')
star.get_data_from(data)
star.get_model_atmosphere()

species_ids = ['CI', 'BaII']
q2.abundances.one(star, species_ids)

print star.CI['ab']
print star.BaII['ab']

import numpy as np
print "Average C and Ba abundances:"
print "A(C)  = {0:.2f} +/- {1:.2f}".      format(np.mean(star.CI['ab']), np.std(star.CI['ab']))
print "A(Ba) = {0:.2f} +/- {1:.2f}".      format(np.mean(star.BaII['ab']), np.std(star.BaII['ab']))

species_ids = ['OI']
q2.abundances.one(star, species_ids, silent=False)

ref = q2.Star('vestaOct')
ref.get_data_from(data)
ref.get_model_atmosphere()

species_ids = ['CI']
q2.abundances.one(star, species_ids, Ref=ref, silent=False, errors=True)

q2.abundances.one(star, Ref=ref, silent=False, errors=True)

species_ids = ['CI', 'OI']
q2.abundances.all(data, 'hyades_co_abundances.csv',
                  species_ids, reference='vestaOct')

q2.abundances.all(data, 'hyades_co_abundances_err.csv',
                  species_ids, 'vestaOct', errors=True)

q2.abundances.all(data, 'hyades_abundances_err.csv',
                  reference='vestaOct', errors=True)

