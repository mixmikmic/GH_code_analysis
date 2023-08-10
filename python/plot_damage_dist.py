get_ipython().magic('matplotlib inline')
from rmtk.plotting.damage_dist import plot_damage_dist as plotdd
tax_dmg_dist_file = '../sample_outputs/scenario_damage/dmg_dist_per_taxonomy.xml'

taxonomy_list=[]
plot_3d = True
plot_no_damage = False

plotdd.plot_taxonomy_damage_dist(tax_dmg_dist_file, taxonomy_list, plot_no_damage, plot_3d)

total_dmg_dist_file = '../sample_outputs/scenario_damage/dmg_dist_total.xml'
plot_no_damage = False

plotdd.plot_total_damage_dist(total_dmg_dist_file, plot_no_damage)

