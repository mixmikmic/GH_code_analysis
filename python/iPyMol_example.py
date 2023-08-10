get_ipython().magic('pylab inline')
from ipymol import viewer as pymol

pymol.start()

pymol.do('fetch 3uex, struct, async=0;')
pymol.do('remove solvent; ')

# smoothing parameters
pymol.do('alter all, b=10; alter all, q=1; set gaussian_resolution, 7.6;')
pymol.do('map_new map, gaussian, 1, n. C+O+N+CA, 5; isosurface surf, map, 1.5;')

# color struct according to atom count
pymol.do('spectrum count, rainbow, struct')

# color the map based on the b-factors of the underlying protein
pymol.do('cmd.ramp_new("ramp", "struct", [0,10,10], "rainbow") ')
 
# set the surface color
pymol.do('cmd.set("surface_color", "ramp", "surf")')
 
# hide the ramp and lines
pymol.do('disable ramp; hide lines;')

pymol.do('show sticks, org; show spheres, org;')

pymol.do('color magenta, org')
pymol.do('set_bond stick_radius, 0.13, org; set sphere_scale, 0.26, org;')
 
pymol.do('set_bond stick_radius, 0.13, org; set_bond stick_color, white, org; set sphere_scale, 0.26, org;')

pymol.do('set_view (    -0.877680123,    0.456324875,   -0.146428943,    0.149618521,   -0.029365506,   -0.988305628,    -0.455291569,   -0.889327347,   -0.042500813,    -0.000035629,    0.000030629,  -37.112102509,    -3.300258160,    6.586110592,   22.637466431,     8.231912613,   65.999290466,  -50.000000000 );')

pymol.do('ray;')

pymol.show()
pymol.do('png png0.png;')

