#Start by importing the profile density module.
import mda_density_profile as dp

#Import MDAnalysis
import MDAnalysis as mda

get_ipython().magic('matplotlib inline')

#Import the plotting functions.
import plot_generation_functions as pgf

#Build the MDAnalysis universe with our topology and trajectory file.
u = mda.Universe('./test_system/mem2.psf', './test_system/mem2_1_fr10.dcd')

print u

#Let's start with bilayer itself.
bilayer = u.select_atoms("not resname CLA and not resname TIP3 and not resname POT")

#Now lets select the water.
water = u.select_atoms('resname TIP3')

# Compute the mass density profile for the bilayer.
dens_prof_bilayer = dp.MassDensityProfile(u.trajectory, bilayer)

# Now let's compute mass density profile for water.
dens_prof_water = dp.MassDensityProfile(u.trajectory, water)

#Now we should plot our results.
pgf.plot_density_profile([dens_prof_water, dens_prof_bilayer], save=False, show=True, label_list=['Water', 'Bilayer'])

# Compute the mass density profile for the bilayer. However, let's give a reference selection (i.e. the bilayer).
dens_prof_bilayer_centered = dp.MassDensityProfile(u.trajectory, bilayer, refsel=bilayer)

# Now let's compute mass density profile for water centered on the bilayer.
dens_prof_water_centered = dp.MassDensityProfile(u.trajectory, water, refsel=bilayer)

#Now we can plot our new results.
pgf.plot_density_profile([dens_prof_water_centered, dens_prof_bilayer_centered], save=False, show=True, label_list=['Water', 'Bilayer'])

#Let's compute the point charge density for the bilayer.
edens_prof_bilayer = dp.ElectronDensityProfile(u.trajectory, bilayer, refsel=bilayer)

#Now water.
edens_prof_water = dp.ElectronDensityProfile(u.trajectory, water, refsel=bilayer)

#Now let's plot the profile.
pgf.plot_density_profile([edens_prof_water, edens_prof_bilayer], save=False, show=True, label_list=['Water', 'Bilayer'], ylabel='Electron Density')

#Let's compute the point charge density for the bilayer.
edens_prof_bilayer_gauss = dp.ElectronDensityProfile_gaussians(u.trajectory, bilayer, refsel=bilayer)
#Now water.
edens_prof_water_gauss = dp.ElectronDensityProfile_gaussians(u.trajectory, water, refsel=bilayer)
#Now let's plot the profile.
pgf.plot_density_profile([edens_prof_water_gauss, edens_prof_bilayer_gauss], save=False, show=True, label_list=['Water', 'Bilayer'], ylabel='Electron Density - Gaussians')



