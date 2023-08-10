# densities in kg/liter
rhoMCHM = 0.9074
rhoW = 1.0

# Mass flows in kg/sec
mRiver = 2650*28.31*rhoW
mMCHM = 7500*3.785*rhoMCHM/(4*3600)

# concentration in ppm by mass
cMCHM = mMCHM*1e6/(mRiver+mMCHM)
print("MCHM Concentration = {:0.1f} ppm".format(cMCHM))

# densities in kg/liter
rhoMCHM = 0.9074
rhoW = 1.0

# Mass flows in kg/sec
mRiver = 271*28.31*rhoW
mMCHM = 7500*3.785*rhoMCHM/(4*3600)

# concentration in ppm by mass
cMCHM = mMCHM*1e6/(mRiver+mMCHM)
print("MCHM Concentration = {:0.1f} ppm".format(cMCHM))

# densities in kg/liter
rhoMCHM = 0.9074
rhoW = 1.0

# Mass flows in kg/sec
mRiver = 271*28.31*rhoW
mMCHM = 0.1e-6*mRiver

# Volumetric flow in liter/hour
vMCHM = 3600*mMCHM/rhoMCHM
print("MCHM Flowrate = {:0.1f} liters/hour".format(vMCHM))



