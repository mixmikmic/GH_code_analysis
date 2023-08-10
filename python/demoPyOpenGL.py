from runPyOpenGL import runPyOpenGL
import numpy as np

code = runPyOpenGL()

np.random.seed(seed = 4444)

n = 2
if (n > 0):
    fac = 0.1
    rfac = 0.1
    zpos = -0.5
    code.objX = np.append(np.random.normal(size = n, loc = 0., scale = fac), 0.)
    code.objY = np.append(np.random.normal(size = n, loc = 0., scale = fac), 0.)
    code.objZ = np.append(np.random.normal(size = n, loc = zpos, scale = fac), zpos)
    code.radius = np.append(np.random.random(size = n)*rfac, rfac)
    code.color = np.append([[r, g, b, 1.] for (r,g,b) in zip(np.random.random(size = n), np.random.random(size = n), np.random.random(size = n))] , [[1.,1.,1.,1.]], axis = 0)
    code.centerX = 0.
    code.centerY = 0.
    code.centerZ = zpos

# Most simple shaders
code.vertexfile = 'shaders/vertex.glsl' 
code.fragmentfile = 'shaders/fragment.glsl'

# This vertex shader includes Lighting
#code.vertexfile = 'shaders/vertexLighting.glsl'

# Either of these fragment shaders can be used with billboards 
#code.fragmentfile = 'shaders/fragmentLimbDarkening.glsl'

#code.fragmentfile = 'shaders/fragmentCircle.glsl'
# if you want transparency to work correctly, you need to sort by the Z location before drawing
#code.doSort = True #[False]

# This shader will produce a Jupiter-like planet, with the texture stored in imfile
#code.fragmentfile = 'shaders/fragmentJupiter.glsl'
#code.imfile = 'shaders/gas_giant_lookup1.png'
#code.time = 0.1 #set this to something > 0 to see changes in Jupiters "clouds"

# Choose one of these drawing types 
code.doSphere = True #[True] Isosheres: looks nice for a few spheres, but gets prohibitively slow for very many spheres)

code.doBillboard = False #[False] Billboards: faster, but can't use lighting in the same way as Isospheres

# let's first try this with the lines drawn.  Comment this out (or set to False) to fill in the spheres
code.drawLine = True #[False]


# do we want to add lighting? (if True, use the correct vertex shader, above)
code.doLighting = False #[False]
code.ambient = [0.1, 0.1, 0.1, 1]
code.diffuse = [0.7, 0.7, 0.7, 1]

code.mainLoop()

