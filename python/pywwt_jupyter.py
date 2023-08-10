from pywwt.jupyter import WWTJupyterWidget

wwt = WWTJupyterWidget()
wwt

from astropy import units as u
from astropy.coordinates import SkyCoord
M31 = SkyCoord('00h42m44.330s +41d16m07.50s')
wwt.center_on_coordinates(M31, fov=10 * u.deg)
wwt.add_circle(M31, radius=1 * u.deg)
wwt.constellation_figures = True

