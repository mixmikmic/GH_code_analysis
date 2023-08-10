from IPython.display import Image

get_ipython().run_cell_magic('writefile', 'ipython_ncl.ncl', 'begin\n\n  f = addfile("$HOME/NCL/NUG/Version_1.0/data/rectilinear_grid_2D.nc","r")\n  printVarSummary(f)\n\n\n  t = f->tsurf\n  printVarSummary(t)\n\n  wks_type            = "png"\n  wks_type@wkWidth    = 800\n  wks_type@wkHeight   = 800\n  wks = gsn_open_wks(wks_type,"plot_contour")\n\n  res                 =  True\n  res@gsnMaximize     =  True\n  res@cnFillOn        =  True\n  res@cnLevelSpacingF =  5\n  res@tiMainString    = "Title string"\n\n  plot = gsn_csm_contour_map(wks,t(1,:,:),res)\n\nend')

get_ipython().system('ncl ipython_ncl.ncl > log')
get_ipython().system('convert -trim +repage plot_contour.png plot_contour_small.png')
Image('plot_contour_small.png')



