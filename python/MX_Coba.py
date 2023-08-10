import travelmaps2 as tm
tm.setup(dpi=200)

fig_x = tm.plt.figure(figsize=(tm.cm2in([11, 6])))

# Locations
MDF = [19.433333, -99.133333] # Mexico City
COB = [20.494722, -87.736111] # Cobá

# Create basemap
m_x = tm.Basemap(width=3500000, height=2300000, resolution='c',  projection='tmerc', lat_0=24, lon_0=-102)

# Plot image
m_x.warpimage('./data/TravelMap/HYP_HR_SR_OB_DR/HYP_HR_SR_OB_DR.tif')

# Put a shade over non-Mexican countries
countries = ['USA', 'BLZ', 'GTM', 'HND', 'SLV', 'NIC', 'CUB']
tm.country(countries, m_x, fc='.8', ec='.3', lw=.5, alpha=.6)

# Fill states
fcs = 32*['none']
ecs = 32*['k']
lws = 32*[.2,]
tm.country('MEX', bmap=m_x, fc=fcs, ec=ecs, lw=lws, adm=1)
ecs = 32*['none']
ecs[22] = 'r'
lws = 32*[1,]
tm.country('MEX', bmap=m_x, fc=fcs, ec=ecs, lw=lws, adm=1)

# Add visited cities
tm.city(COB, 'Coba', m_x, offs=[-1, 0], halign="right")
tm.city(MDF, 'Mexiko-Stadt', m_x, offs=[-.6, .6], halign="right")

# Save-path
#fpath = '../mexico.werthmuller.org/content/images/coba/'
#tm.plt.savefig(fpath+'MapCoba.png', bbox_inches='tight')
tm.plt.show()

