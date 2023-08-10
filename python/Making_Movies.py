s=ms.star_log(mass=2,Z=0.02)

s.kippenhahn_CO(1,'model')

s.kip_cont(modstart=26100,modstop=28000,ylims=[0.54,0.6],engenPlus=False)

pt=mp.se(mass=2,Z=0.02)

cycs=range(26100,27100,20)

species=['H-1','He-4','C-12','C-13','N-14','O-16','Ba-138']
pt.movie(cycs,plotstyle='plot',x_item='mass',
         y_items=species,logy=True,xlims=(0.575,0.579),
         ylims=(-10,0.),interval=100,legend=True,loc='lower right')

pt.movie(cycs,plotstyle='iso_abund',
         amass_range=[50,160],mass_range=[0.5765,0.5770])

pt.movie(cycs,plotstyle='abu_chart',
                mass_range=[0.576,0.578],plotaxis=[0, 80, 0, 60],
                ilabel=False,imlabel=False,boxstable=False)

pt.abund_at_masscoorinate(26100,0.578,online=True)



