# Import Python libraries to work with SciServer
import SciServer.CasJobs as CasJobs # query with CasJobs
import SciServer.SkyServer as SkyServer # look up objects and generate images
print('SciServer libraries imported')

# Import some other needed libraries
import pandas
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
print('Other libraries imported')

# Apply some settings

# ensure columns get written completely in notebook
pandas.set_option('display.max_colwidth', -1)
# do *not* show python warnings 
import warnings
warnings.filterwarnings('ignore')
print('Settings applied')

## Create a query to fetch stars for one of the following globular clusters
# Pal 3: RA = 151.3801, Dec = +0.072
# Pal 5: RA = 229.0128, Dec = -0.1082 

# Enter the coordinates of your star cluster (to be used to construct the query)
cluster_name = 'Palomar 5'
cluster_ra = 229.0128  # RA of the center of the cluster (in decimal degrees)
cluster_dec = -0.182   # Dec of the center of the cluster (in decimal degrees)
cluster_radius = 4.0   # Radius of the cluster, in arcminutes
cluster_distance = 23000 # approximate distance to your cluster, in units of parsecs

query = 'SELECT p.objid, p.psfMag_u AS u, p.psfMagErr_u as u_err, p.psfMag_g AS g, p.psfMagErr_g as g_err, '
#query = 'SELECT p.objid, p.dered_u AS u, p.psfMagErr_u as u_err, p.dered_g AS g, p.psfMagErr_g as g_err, ' # de-reddened mags
query += 'p.dered_r AS r, p.psfMagErr_r as r_err, p.dered_i AS i, p.psfMagErr_i as i_err, '
query += 'p.dered_z AS z, p.psfMagErr_z as z_err '
query += 'FROM photoObj p '
query += 'JOIN dbo.fGetNearbyObjEq(' + str(cluster_ra) + ', ' + str(cluster_dec) + ', ' + str(cluster_radius) + ') n '
query += 'ON p.objid = n.objID '
query += 'WHERE p.type = 6 '
query += 'AND p.psfMag_g BETWEEN -10 AND 23 '
query += 'AND p.psfMag_u/p.psfMagerr_u > 30 '
query += 'AND p.psfMag_g/p.psfMagerr_g > 30 '
query += 'AND p.psfMag_r/p.psfMagerr_r > 30 '
query += 'AND p.psfMag_i/p.psfMagerr_i > 30 '
query += 'AND p.psfMag_z/p.psfMagerr_z > 30'
#print(query)   # useful for debugging, remove first # to uncomment

# send query to CasJobs
stars = CasJobs.executeQuery(query, "dr14")

stars

color = stars['g']-stars['r']
color_err = np.sqrt((stars['g_err']**2) + (stars['r_err']**2))


#plt.scatter(color,stars['r'], alpha=0.1)

#fig, ax = plt.subplots()
#ax.errorbar(x, y, xerr=0.2, yerr=0.4)
#plt.show()
#plt.figure(figsize=(9,6))
plt.errorbar(color, stars['r'].values, yerr = color_err, xerr=stars['r_err'].values, fmt='mo', alpha=0.2, label=cluster_name)

plt.axis([-1,2,24,10])
plt.legend(numpoints=1)
plt.xlabel('g - r')
plt.ylabel('r')
plt.show()

color = stars['g']-stars['r']
color_err = stars['g_err']+stars['r_err']

#plt.scatter(color,stars['r'], alpha=0.1)
plt.errorbar(color, stars['g'], yerr = color_err, xerr=stars['r_err'], fmt='mo', alpha=0.2, label=cluster_name)
plt.axis([-1,2,24,10])
plt.legend(numpoints=1)
plt.xlabel('g - r')
plt.ylabel('g')
plt.annotate('RGB', xy=(0.6, 19), xytext=(1.0, 18),
            arrowprops=dict(facecolor='gray', shrink=0.05))
plt.annotate('Main Sequence', xy=(0.5, 22), xytext=(0.8, 21),
            arrowprops=dict(facecolor='gray', shrink=0.03))
plt.annotate('Horiz. Branch', xy=(0.2, 17), xytext=(-0.8, 16),
            arrowprops=dict(facecolor='gray', shrink=0.03))
plt.annotate('MS Turnoff', xy=(0.2, 20.5), xytext=(-0.8, 20),
            arrowprops=dict(facecolor='gray', shrink=0.03))
plt.show() 

## Provide the name of your isochrome model file below:

isofile = "isochrones/tmp1487212885.iso"  # [Fe/H]=-2.0
#isofile = "tmp1487281984.iso" # [Fe/H]=0.2
print('Isochrone file loaded: '+isofile)

## Prepared for you below is a function that will 
## read in the models from your uploaded isochrome file.
def read_isofile(filename):

    # Set a minimum and maximum age range (in Gyrs) below:
    minage = 11.0
    maxage = 12.0
    sdss_u = []
    sdss_g = []
    sdss_r = []
    sdss_i = []
    sdss_z = []

    age = [] # in Gigayears
    mass = [] # ratio of stellar mass to M_sun
    logT = [] # log of the effective Temperature (K)
    logL = [] # log of the ratio of stellar luminosity to L_sun
    counter=0
    for line in open(filename).readlines():
        counter+=1
        cols = line.split()
        if (line.startswith('#')):
    #        print(counter, ': starts with hashtag: ',cols)
            if (counter >= 8):   # real data starts on line #8
                if ("AGE" in cols[0]):
                    block = (line[0:11])
                    thisage = float(block.split('=')[1])
    #                print("Reading stellar models with age: {0:.2f} Gyr".format(thisage))            
        else:
            if (thisage > minage) and (thisage < maxage):
#                print(cols)
                if (len(cols) > 0):
                    age.append(thisage)
                    mass.append(float(cols[1]))
                    logT.append(float(cols[2]))
                    logL.append(float(cols[4]))
                    sdss_u.append(float(cols[5]))
                    sdss_g.append(float(cols[6]))
                    sdss_r.append(float(cols[7]))
                    sdss_i.append(float(cols[8]))
                    sdss_z.append(float(cols[9]))
    return (sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, age, mass, logT, logL)    

(sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, age, mass, logT, logL) = read_isofile(isofile)

print("Read in {0:.0f} stellar models, ranging in age from {1:.2f} to {2:.2f} Gyr.".format(len(sdss_u), age[0], age[-1]))

model_color = np.array(sdss_g)-np.array(sdss_r)
model_g = np.array(sdss_g)


#plt.scatter(color,stars['r'], alpha=0.1)
plt.errorbar(color, stars['g'], yerr = color_err, xerr=stars['r_err'], fmt='mo', alpha=0.2, label='Palomar 5')
plt.axis([-1,2,24,-2])
plt.legend(numpoints=1)
plt.xlabel('g - r')
plt.ylabel('g')
plt.plot(model_color,model_g, 'b.', alpha=0.02)
plt.show()

from math import log10
#distance = 23000.0 # approximate distance to Palomar 5 in units of parsecs

dm = 5*log10(cluster_distance/10)

model_g_corr = model_g+dm

plt.errorbar(color, stars['g'], yerr = color_err, xerr=stars['r_err'], fmt='mo', alpha=0.2, label='Palomar 5')
plt.axis([-1,2,24,10])
plt.legend(numpoints=1)
plt.xlabel('g - r')
plt.ylabel('g')
plt.plot(model_color,model_g_corr, 'b.', alpha=0.2)
plt.show()


ebv = 0.1
av = 3.2*ebv

model_g_dered = np.array(model_g_corr) + av
model_gr_dered = np.array(model_color) + ebv
print('Done')

#plt.figure(figsize=(12,8))
plt.errorbar(color, stars['g'], yerr = color_err, xerr=stars['r_err'], fmt='mo', alpha=0.2, label='Palomar 5')
plt.axis([-1,2,24,10])
plt.legend(numpoints=1)
plt.xlabel('g - r')
plt.ylabel('g')
plt.plot(model_gr_dered,model_g_dered, 'b.', alpha=0.2)
plt.show()

## Create a query to fetch stars for one of the following globular clusters
# Pal 3: RA = 151.3801, Dec = +0.072
# Pal 5: RA = 229.0128, Dec = -0.1082 

# Enter the coordinates of your star cluster (to be used to construct the query)
cluster_name = 'Palomar 5'
cluster_ra = 229.0128  # RA of the center of the cluster (in decimal degrees)
cluster_dec = -0.182   # Dec of the center of the cluster (in decimal degrees)
cluster_radius = 4.0   # Radius of the cluster, in arcminutes
cluster_distance = 23000 # approximate distance to your cluster, in units of parsecs

query_psf = 'SELECT p.objid, p.psfMag_u AS u, p.psfMagErr_u as u_err, p.psfMag_g AS g, p.psfMagErr_g as g_err, '
query_psf += 'p.psfMag_r AS r, p.psfMagErr_r as r_err, p.psfMag_i AS i, p.psfMagErr_i as i_err, '
query_psf += 'p.psfMag_z AS z, p.psfMagErr_z as z_err '
query_psf += 'FROM photoObj p '
query_psf += 'JOIN dbo.fGetNearbyObjEq(' + str(cluster_ra) + ', ' + str(cluster_dec) + ', ' + str(cluster_radius) + ') n '
query_psf += 'ON p.objid = n.objID '
query_psf += 'WHERE p.type = 6 '
query_psf += 'AND p.psfMag_g BETWEEN -10 AND 23 '
query_psf += 'AND p.psfMag_u/p.psfMagerr_u > 30 '
query_psf += 'AND p.psfMag_g/p.psfMagerr_g > 30 '
query_psf += 'AND p.psfMag_r/p.psfMagerr_r > 30 '
query_psf += 'AND p.psfMag_i/p.psfMagerr_i > 30 '
query_psf += 'AND p.psfMag_z/p.psfMagerr_z > 30'
#print(query)   # useful for debugging, remove first # to uncomment

# send query to CasJobs
stars_psf = CasJobs.executeQuery(query_psf, "dr14")

stars_psf

## Create a query to fetch stars for one of the following globular clusters
# Pal 3: RA = 151.3801, Dec = +0.072
# Pal 5: RA = 229.0128, Dec = -0.1082 

# Enter the coordinates of your star cluster (to be used to construct the query)
cluster_name = 'Palomar 5'
cluster_ra = 229.0128  # RA of the center of the cluster (in decimal degrees)
cluster_dec = -0.182   # Dec of the center of the cluster (in decimal degrees)
cluster_radius = 4.0   # Radius of the cluster, in arcminutes
cluster_distance = 23000 # approximate distance to your cluster, in units of parsecs

query_dered = 'SELECT p.objid, p.dered_u AS u, p.modelMagErr_u as u_err, p.dered_g AS g, p.modelMagErr_g as g_err, '
query_dered += 'p.dered_r AS r, p.modelMagErr_r as r_err, p.dered_i AS i, p.modelMagErr_i as i_err, '
query_dered += 'p.dered_z AS z, p.modelMagErr_z as z_err '
query_dered += 'FROM photoObj p '
query_dered += 'JOIN dbo.fGetNearbyObjEq(' + str(cluster_ra) + ', ' + str(cluster_dec) + ', ' + str(cluster_radius) + ') n '
query_dered += 'ON p.objid = n.objID '
query_dered += 'WHERE p.type = 6 '
query_dered += 'AND p.dered_g BETWEEN -10 AND 23 '
query_dered += 'AND p.dered_u/p.modelMagErr_u > 30 '
query_dered += 'AND p.dered_g/p.modelMagErr_g > 30 '
query_dered += 'AND p.dered_r/p.modelMagErr_r > 30 '
query_dered += 'AND p.dered_i/p.modelMagErr_i > 30 '
query_dered += 'AND p.dered_z/p.modelMagErr_z > 30 '
#print(query)   # useful for debugging, remove first # to uncomment

# send query to CasJobs
stars_dered = CasJobs.executeQuery(query_dered, "dr14")

stars_dered

color_psf = stars_psf['g']-stars_psf['r']
color_err_psf = np.sqrt((stars_psf['g_err']**2) + (stars_psf['r_err']**2))

color_dered = stars_dered['g']-stars_dered['r']
color_err_dered = np.sqrt((stars_dered['g_err']**2) + (stars_dered['r_err']**2))



ebv = 0.1
av = 3.2*ebv

model_g_dered = np.array(model_g_corr) + av
model_gr_dered = np.array(model_color) + ebv
print('Done')




fig = plt.figure(figsize=(18,12))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

#plt.scatter(color,stars['r'], alpha=0.1)

#fig, ax = plt.subplots()
#ax.errorbar(x, y, xerr=0.2, yerr=0.4)
#plt.show()
#plt.figure(figsize=(9,6))

ax1.errorbar(color_psf, stars_psf['r'].values, yerr = color_err_psf, xerr=stars_psf['r_err'].values, fmt='mo', alpha=0.2, label=cluster_name)
ax2.errorbar(color_dered, stars_dered['r'].values, yerr = color_err_dered, xerr=stars_dered['r_err'].values, fmt='mo', alpha=0.2, label=cluster_name)


ax1.axis([-1,2,24,10])

ax2.axis([-1,2,24,10])

#ax1.plot(model_color,model_g_corr, 'b.', alpha=0.2)
ax1.plot(model_gr_dered,model_g_dered, 'b.', alpha=0.2)
ax2.plot(model_gr_dered,model_g_dered, 'b.', alpha=0.2)

ax1.set_title('PSF Magnitudes')
ax1.set_xlabel('g-r')
ax1.set_ylabel('r')

ax2.set_title('Dereddened model magnitudes')
ax2.set_xlabel('g-r')
ax2.set_ylabel('r')

plt.xlabel('g - r')
plt.ylabel('r')
plt.legend(numpoints=1)


plt.show() 



