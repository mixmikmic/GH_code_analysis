get_ipython().run_line_magic('pylab', 'inline')
import pysd
import numpy as np
import pandas as pd

model = pysd.read_vensim('bass_diffusion.mdl')
model.doc()

total_potential = model.components.potential_adopters_p()

model.set_components({'Advertising Effectiveness a':0})

collector = []
for f_adopt in np.arange(0,1.01,.025):
    model.set_initial_condition((0, {'Adopters A': f_adopt*total_potential, 
                                     'Potential Adopters P': (1-f_adopt)*total_potential}))
    collector.append({'Fraction Having Adopted': f_adopt, 'Adoption Rate': model.components.adoption_rate_ar()})
    
result = pd.DataFrame(collector)
result.set_index('Fraction Having Adopted', inplace=True)
result.plot(fontsize=16, legend=False, linewidth=0, marker='o')
plt.yticks([])
plt.xticks([0, .25, .5, .75, 1])
plt.xlabel('Fraction Having Adopted', fontsize=16)
plt.ylabel('Adoption Rate', fontsize=16)
plt.grid('on')
plt.ylim(0, 35)
plt.savefig('Phase_Plane_noad.png')

model.set_components({'Advertising Effectiveness a':0.05})

collector = []
for f_adopt in np.arange(0,1.01,.025):
    model.set_initial_condition((0, {'Adopters A': f_adopt*total_potential, 
                                     'Potential Adopters P': (1-f_adopt)*total_potential}))
    collector.append({'Fraction Having Adopted': f_adopt, 'Adoption Rate': model.components.adoption_rate_ar()})
    
result = pd.DataFrame(collector)
result.set_index('Fraction Having Adopted', inplace=True)
result.plot(fontsize=16, legend=False, linewidth=0, marker='o')
plt.yticks([])
plt.xticks([0, .25, .5, .75, 1])
plt.xlabel('Fraction Having Adopted', fontsize=16)
plt.ylabel('Adoption Rate', fontsize=16)
plt.grid('on')
#plt.ylim(0, 35)
plt.savefig('Phase_Plane_noad.png')



