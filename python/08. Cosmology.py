from astropy.cosmology import Planck15 as planck

planck.h

planck.comoving_distance([0.5, 1.0, 1.5])

planck.comoving_volume([0.5,1.0,1.5])

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(name='my_cosmo', H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
cosmo

cosmo.age(0) 

from astropy.cosmology import Planck13, z_at_value

z_at_value(Planck13.age, 10 * u.Gyr) 

