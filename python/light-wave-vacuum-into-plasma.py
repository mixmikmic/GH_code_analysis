import osiris as osiris
get_ipython().run_line_magic('matplotlib', 'inline')

dirname = 'wall'
osiris.runosiris(rundir=dirname,inputfile='wall.txt')

dirname = 'wall-over'
osiris.runosiris(rundir=dirname,inputfile='wall-over.txt')

dirname = 'grad'
osiris.runosiris(rundir=dirname,inputfile='grad.txt')

dirname = 'grad-over'
osiris.runosiris(rundir=dirname,inputfile='grad-over.txt')

dirname = 'wall'
osiris.phasespace(rundir=dirname,dataset='x1',time=0,xlim=[0,70])

dirname = 'wall'
osiris.plot_tx(rundir=dirname, plot_or=3, xlim=[0,70], tlim=[0,150], vmin=-0.01,vmax=0.01, cmap='jet',
              show_cutoff=True)

dirname = 'wall'
osiris.plot_tx(rundir=dirname, plot_or=3, xlim=[27,33], tlim=[0,150], vmin=-0.01,vmax=0.01, cmap='jet',
              show_cutoff=True)

dirname = 'grad'
osiris.phasespace(rundir=dirname,dataset='x1',time=0,xlim=[0,70])

dirname = 'grad'
osiris.plot_tx(rundir=dirname, b0_mag=0.0, plot_or=3, xlim=[0,70], tlim=[0,160], vmin=-0.01,vmax=0.01, cmap='jet',
              show_cutoff=True)

dirname = 'wall-over'
osiris.phasespace(rundir=dirname,dataset='x1',time=0,xlim=[0,70])

dirname = 'wall-over'
osiris.plot_tx(rundir=dirname, plot_or=3, xlim=[0,70], tlim=[0,160], vmin=-0.02,vmax=0.02, cmap='jet',
              show_cutoff=True)

dirname = 'grad-over'
osiris.phasespace(rundir=dirname,dataset='x1',time=0,xlim=[0,70])

dirname = 'grad-over'
osiris.plot_tx(rundir=dirname, b0_mag=0.0, plot_or=3, xlim=[0,70], tlim=[0,160], vmin=-0.02,vmax=0.02, cmap='jet',
              show_cutoff=True)

