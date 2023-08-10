from astropy.coordinates import get_sun

d = get_sun(Time.now()).distance
delta_t = d/c
delta_t.to(u.min)

from astropy.constants import G, M_sun, M_earth

# Force of gravity from the Sun
m1 = 60*u.kg
F_sun = G * m1 * M_sun / d**2

# Calculate mass of neutron star ball
rho = 3.7e17 * u.kg/u.m**3
r_bowlingball = 22 * u.cm
volume = 4./3 * np.pi * r_bowlingball**3
m_bowlingball = rho * volume

# Force of gravity from the neutron star ball
d_bowlingball = 12*u.km

F_bowlingball = G * m1 * m_bowlingball / d_bowlingball**2

# Which is greater?
F_bowlingball > F_sun

from astropy.units import R_sun

# Schwarzschild radius:
r_s = 2 * G * 4.31e6 * M_sun / c**2
print("Schwarzschild radius = {0}".format(r_s.to(R_sun)))

# Size on the sky given small angle approximation
sgr_a_distance = 7940*u.pc
angular_diameter = np.arctan(2*r_s / sgr_a_distance)
angular_diameter.to(u.uarcsec)

birthday = Time('2000-02-21 10:00:00', format='iso')
formats = ['iso', 'jd', 'mjd', 'decimalyear']

for fmt in formats:
    print(getattr(birthday, fmt))

brightest_index = np.argmin(landolt_table['Vmag'])

name = landolt_table['Star'][brightest_index]
ra = landolt_table['_RAJ2000'].quantity[brightest_index]
dec = landolt_table['_DEJ2000'].quantity[brightest_index]

print(name, SkyCoord(ra=ra, dec=dec).galactic)



