import q2

data = q2.Data('standards_stars.csv', 'standards_lines.csv')
arcturus = q2.Star('Arcturus')
arcturus.get_data_from(data)

get_ipython().magic('matplotlib inline')

sp = q2.specpars.SolvePars()
sp.step_teff = 8
sp.step_logg = 0.08
sp.step_vt = 0.08
sp.niter = 35
sp.grid = 'marcs'

pp = q2.specpars.PlotPars()
pp.afe = [-1000, 0.25]

q2.specpars.solve_one(arcturus, sp, PlotPars=pp)

sun = q2.Star('Sun')
sun.get_data_from(data)

q2.specpars.solve_one(arcturus, sp, Ref=sun, PlotPars=pp)

print "[Fe/H](Fe I)  = {0:5.3f} +/- {1:5.3f}".      format(arcturus.iron_stats['afe1'], arcturus.iron_stats['err_afe1'])
print "[Fe/H](Fe II) = {0:5.3f} +/- {1:5.3f}".      format(arcturus.iron_stats['afe2'], arcturus.iron_stats['err_afe2'])
print "A(FeI) vs. EP slope  = {0:.6f}".format(arcturus.iron_stats['slope_ep'])
print "A(FeI) vs. REW slope = {0:.6f}".format(arcturus.iron_stats['slope_rew'])

print "Final stellar parameters:"
print "Teff = {0:4.0f} K, logg = {1:4.2f}, [Fe/H]= {2:5.2f}, vt = {3:4.2f} km/s".      format(arcturus.teff, arcturus.logg, arcturus.feh, arcturus.vt)

print ""
print arcturus

q2.errors.error_one(arcturus, sp, Ref=sun)
print "err_Teff = {0:2.0f} K, err_logg = {1:4.2f}, err_[Fe/H] = {2:4.2f}, err_vt = {3:4.2f}".      format(arcturus.sp_err['teff'], arcturus.sp_err['logg'], arcturus.sp_err['afe'], arcturus.sp_err['vt'])

