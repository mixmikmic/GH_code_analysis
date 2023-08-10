get_ipython().run_cell_magic('file', 'test.f90', '\nprogram example\n  use netcdf\n\n  integer, parameter :: n_time = 366\n  integer, parameter :: n_lon = 720\n  integer, parameter :: n_lat = 360\n  real(kind=4), dimension(n_lon, n_lat, n_time) :: outflow\n  character(len=*), parameter :: unit = \'m3 s-1\'\n  ! this is an example dataset from Dai Yamazaki\n  character(len=*), parameter :: infile = \'/Users/baart_f/data/boulder/dai/hessel/outflw2000.bin\'\n  character(len=*), parameter :: standard_name = \'water_volume_transport_into_sea_water_from_rivers\'\n  ! Better is to use a newunit function or newunit option for newer fortrans\n  integer :: infile_unit = 100\n  ! for looping\n  integer :: i, status\n  integer :: ncid\n  integer :: latdim_id, londim_id, timedim_id\n  integer :: outflowvar_id\n  ! not your everyday fill value (apprx 1e20)\n  real(kind=4) :: fill_value = 100000002004087734272.0\n\n  ! Read the input file in unformatted stream form, as big_endian.\n  open(unit=infile_unit, &\n       file=trim(infile), &\n       form=\'unformatted\', &\n       access=\'stream\', &\n       iostat=status)\n  ! not needed                                                                                                          \n  ! convert=\'big_endian\' \n  read(infile_unit) outflow\n  close(infile_unit)\n\n  ! create a new file\n  status = nf90_create( &\n       path="foo.nc", &\n       cmode=ior(NF90_CLOBBER,NF90_HDF5), &\n       ncid=ncid &\n       )\n    \n  ! define the dimensions\n  status = nf90_def_dim(ncid, "lat", n_lat, latdim_id)\n  status = nf90_def_dim(ncid, "lon", n_lon, londim_id)\n  status = nf90_def_dim(ncid, "time", n_time, timedim_id)\n    \n  ! define the variables\n  status = nf90_def_var(ncid, "outflow", nf90_float, &\n       dimids=(/ londim_id, latdim_id, timedim_id /), &\n       deflate_level=5, &\n       varid=outflowvar_id)\n  \n  ! add some attributes on the file\n  status = nf90_put_att(ncid, 0, \'title\', \'Outflow 2000 dataset\')\n  status = nf90_put_att(ncid, 0, \'institution\', \'Japan Agency for Marine-Earth Science and Technology\')\n  status = nf90_put_att(ncid, 0, \'source\', \'CaMa Flood model\')\n  status = nf90_put_att(ncid, 0, \'resolution\', \'0.5 degrees\')\n  status = nf90_put_att(ncid, 0, \'Conventions\', \'CF-1.6\')\n  status = nf90_put_att(ncid, 0, \'history\', \'created with test.f90\')\n\n  ! add some attributes to the variable\n  status = nf90_put_att(ncid, outflowvar_id, \'_FillValue\', fill_value)\n  status = nf90_put_att(ncid, outflowvar_id, \'units\', unit)\n  status = nf90_put_att(ncid, outflowvar_id, \'standard_name\', standard_name)\n    \n  ! we have defined the variables  \n  status = nf90_enddef(ncid)\n    \n  ! now we can write the data\n  status = nf90_put_var(ncid, outflowvar_id, outflow)\n    \n  ! and done\n  status = nf90_close(ncid=ncid)\n\n\nend program example')

get_ipython().run_cell_magic('bash', '', 'gfortran -c -ffree-line-length-none -ffree-form -I/opt/local/include test.f90\ngfortran -L/opt/local/lib test.o -lnetcdf -lnetcdff -o test\n')

get_ipython().run_cell_magic('bash', '', './test\nncdump -h foo.nc\nls -alh foo.nc  /Users/baart_f/data/boulder/dai/hessel/outflw2000.bin')

import numpy as np
import netCDF4
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

var = netCDF4.Dataset('foo.nc').variables['outflow']

var.dimensions
var[0,0,0]

fig, ax = plt.subplots(figsize=(13,8))
im = ax.imshow(np.log10(var[0]), cmap='Blues')
name = var.standard_name.replace('_', ' ').capitalize()
unit = var.units
title = "{name} [{unit}] (log)".format(**locals()) 
ax.set_title(title)
plt.colorbar(im, ax=ax)

