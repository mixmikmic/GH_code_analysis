get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
import healpy as hp
import surveyStatus as ss
import copy

field_data = np.loadtxt('fieldID.dat', delimiter='|', skiprows=1,
                        dtype=zip(['id', 'ra', 'dec'], [int, float, float]))

ra_range = 15.  # Degrees
dec_range = 25.
good = np.where(((field_data['ra'] <= ra_range) | (field_data['ra'] >= 360.-ra_range)) &
                ((field_data['dec'] >= -dec_range) & (field_data['dec'] <= dec_range)))

field_data = field_data[good]
field_data['ra'] = np.radians(field_data['ra'])
field_data['dec'] = np.radians(field_data['dec'])

tracker = ss.countFilterStatus()
hpl = ss.HealpixLookup()

class visit(object):
    def __init__(self):
        self.filter = 'r'

vis = visit()
for obs in field_data:
    pix = hpl.lookup(obs['ra'], obs['dec'])
    tracker.add_visit(vis, pix)

hp.mollview(tracker.survey_map)

def plot_ps(inmap, remove_dipole=True, label=None):
    cl = hp.anafast(hp.remove_dipole(inmap))
    ell = np.arange(np.size(cl))
    if remove_dipole:
        condition = (ell > 1)
    else:
        condition = (ell > 0)
    ell = ell[condition]
    cl = cl[condition]
    plt.plot(ell, (cl * ell * (ell + 1)) / 2.0 / np.pi, label=label)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$l(l+1)C_l/(2\pi)$')
    return cl, ell

cl, ell = plot_ps(tracker.survey_map.astype(float))

ra_shifts = np.radians(np.linspace(-3., 3., 40.))
dec_shifts = np.radians(np.linspace(-3., 3., 40.))

dithered_maps = []
x = []
y= []
for ra_shift in ra_shifts:
    for dec_shift in dec_shifts:
        x.append(ra_shift)
        y.append(dec_shift)
        new_map = ss.countFilterStatus()
        for obs in field_data:
            pix = hpl.lookup(obs['ra'], obs['dec'])
            new_map.add_visit(vis, pix)
            pix = hpl.lookup(obs['ra']+ra_shift, obs['dec']+dec_shift)
            new_map.add_visit(vis, pix)
        new_map.survey_map[np.where(new_map.survey_map == 0)] = hp.UNSEEN
        dithered_maps.append(new_map)

hp.mollview(dithered_maps[0].survey_map)
hp.mollview(dithered_maps[10].survey_map)
hp.mollview(dithered_maps[20].survey_map)

cl, ell = plot_ps(dithered_maps[0].survey_map)
cl, ell = plot_ps(dithered_maps[0].survey_map, remove_dipole=False)

for i in [0,5,10, 26]:
    cl, ell = plot_ps(dithered_maps[i].survey_map, label=str(i))
plt.legend()

ps_s = [np.max(hp.anafast(inmap.survey_map.astype(float))[50:]) for inmap in dithered_maps]
ps_s = np.array(ps_s)

plt.scatter(np.degrees(x)*60., np.degrees(y)*60., c=np.array(ps_s))
cb = plt.colorbar()
plt.xlabel('RA shift (arcmin)')
plt.ylabel('Dec Shift (arcmin)')
cb.set_label('max power')

ps_s = np.array(ps_s)
min_power = np.where(ps_s == ps_s.min())
max_power = np.where(ps_s == ps_s.max())

hp.mollview(dithered_maps[min_power[0]].survey_map, title='minimum power map')
hp.mollview(dithered_maps[max_power[0]].survey_map, title='max power map')

