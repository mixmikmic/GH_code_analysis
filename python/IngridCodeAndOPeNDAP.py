import xarray as xr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

nc_url = "http://iridl.ldeo.columbia.edu/SOURCES/.WORLDBATH432/.bath/dods"
dsURL = xr.open_dataset(nc_url)
dsURL.bath.plot()

ingridsnippet = """
CMIP5 .byScenario .rcp85 .atmos .mon .ts .CCSM4 .r1i1p1 .ts
 dup [T]detrend-bfl sub
 dup T last VALUE exch T first VALUE sub 
(trend)rn
""" 

#now generate the URL, subsituting '/' for ' ':
sss = ' '.join(ingridsnippet.split()).replace(' ','/') #take out extra whitespaces first
nc_url2 = 'http://strega.ldeo.columbia.edu:81/'+sss+'/dods'
print(nc_url2)
xr.open_dataset(nc_url2).trend.plot()

import os.path
if os.path.exists("/usr/local/bin/ingrid"):
    from subprocess import Popen, PIPE

    # generate a multiline string containing the ingrid commands:
    ingridcode = """
\\begin{ingrid}
SOURCES .WORLDBATH432 .bath
(bath.nc)writeCDF 
%% /plotname (bath-from-notebook) def
%% X Y fig: colors coasts :fig .ps
\\end{ingrid}
    """ 

    # and then pipe it to the ingrid executable:
    p = Popen(['/usr/local/bin/ingrid'], stdin=PIPE, stdout=PIPE) 
    ingridout, ingriderr = p.communicate(input=bytes(ingridcode, 'utf-8'))

    # now there will be the netcdf file, 'bath.nc' in your local directory
    nc_file = 'bath.nc'
    dsCODE = xr.open_dataset(nc_file)
    # to see that both give the same result:
    plt.figure(figsize=(8,2))
    plt.subplot(121)
    dsURL.bath.plot()
    plt.subplot(122)
    dsCODE.bath.plot()
    plt.tight_layout()

    # and can check directly:
    assert dsURL.equals(dsCODE)
    import sys
    sys.path.append("/net/carney/home/naomi/mymodules")
    from ingrid.code import callIngrid

    # Ingrid comments should begin with %% to avoid misinterpretation
    # Variable substitution can be done a couple of ways, here is one:
    var = 'bath'
    file = 'bath-another.nc'

    ingridcode = """
    \\begin{ingrid}
    SOURCES .WORLDBATH .%s
    Y 0 60 RANGE X (90W) (0W) RANGE
    (%s)writeCDF 
    %%X Y fig: colors :fig .ps
    \\end{ingrid}
    """  %(var, file)

    callIngrid(ingridcode)
    xr.open_dataset(file).bath.plot(cmap='tab20b',vmin=-6000,vmax=6000)
else:
    print('sorry, no ingrid on your machine')

