import mesa as ms
get_ipython().magic('pylab nbagg')

nugrid_data_dir='/home/nugrid/CADC/NuGrid'
m2_dir=nugrid_data_dir+'/data/teaching/mesa/m2.00z2.0e-2/LOGS'
m20_dir=nugrid_data_dir+'/data/teaching/mesa/m20.0z2.0e-2/LOGS'

s2=ms.star_log(m2_dir)
s20=ms.star_log(m20_dir)

ifig=1; figure(ifig)
s20.hrd_new()
s2.hrd_new()

# what colums are available in this history data instance?
s2.cols

# plot any of these against any other
ifig=2; figure(ifig)     # start new figure
s2.plot('model_number','log_Teff')

# after you found a profile you are interested in, e.g. a fully 
# convective pre-main sequence model you may create a profile instance
# for a model number
p2=ms.mesa_profile(m2_dir,200)

# and again you may ask what columns are available
p2.cols

# let's verify that indeed this model is fully convective by plotting the
# radiative and adiabatic temperature gradient against each other
ifig=3; figure(ifig)
grada=p2.get('grada')
gradr=p2.get('gradr')
mass=p2.get('mass')
plot(mass,log10(grada))
plot(mass,log10(gradr))

# here we may also plot arbitrary quantities against each other
ifig=4; figure(ifig)
p2.plot('mass','logRho')

ifig=5; figure(ifig)
p2.plot('logP','logRho')

# Now integrate a adiabatic polytrope with estimated central conditions for the 
# radius and mass of this stellar model and see if 

