import numpy as np
import bqplot # Bloomberg plotting package with straightforward interactivity
import sectional_tess #package in this repository with sectional tessellation code
#from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d

# Set initial positions of generators

# 5x5 grid
# x_data = np.repeat(np.arange(5),5)
# y_data = np.tile(np.arange(5),5)

# Concentric circles:
# Outernum = #generators along an outer circle of radius 2
# Innernum is #generators along a circle of radius 1
# Default (outernum,innernum)=(3,2) produces a framework roughly resembling the Eiffel Tower
outernum = 3
innernum = 2
y_data = -np.concatenate((2.*np.cos(2.*np.pi/float(outernum)*np.arange(outernum)),
                         1.*np.cos(2.*np.pi*np.arange(innernum)/float(outernum))))+2.
x_data = np.concatenate((2.*np.sin(2.*np.pi/float(outernum)*np.arange(outernum)),
                        1.*np.sin(2.*np.pi*np.arange(innernum)/float(innernum))))+2.

## User interface based on https://githubqplot.com/bloomberg/bqplot/blob/master/examples/Marks/Scatter.ipynb
sc_x = bqplot.LinearScale(stabilized=True,max=5,min=-1)
sc_y = bqplot.LinearScale(stabilized=True,max=5,min=-1)

scat_height = bqplot.Scatter(x=x_data, y=y_data, scales={'x': sc_x, 'y': sc_y}, colors=['green'],
               enable_move=True, restrict_y=True)
scat_height.y_data_init = 1.*y_data
scat = bqplot.Scatter(x=x_data, y=y_data, scales={'x': sc_x, 'y': sc_y}, colors=['blue'],
               enable_move=True)

lin = bqplot.Lines(x=[], y=[], scales={'x': sc_x, 'y': sc_y}, colors=['black'])
lin_ext = bqplot.Lines(x=[], y=[], scales={'x': sc_x, 'y': sc_y}, colors=['black'])

def update_line(change=None):
    with lin.hold_sync():        
        # if a point was added to scat
        if (len(scat.y) == len(scat_height.y) + 1):
            scat_height.y = np.append(scat_height.y, scat.y[-1])
        if (len(scat.y) == len(scat_height.y_data_init) + 1):
            scat_height.y_data_init = np.append(scat_height.y_data_init, scat.y[-1])
        if (len(scat.x) == len(scat_height.x) + 1):
            scat_height.x = np.append(scat_height.x, scat.x[-1])            
            
        # if a point was added to scat_height
        if (len(scat_height.y) == len(scat.y) + 1):
            scat.y = np.append(scat.y, scat_height.y[-1])  
        if (len(scat_height.y) == len(scat_height.y_data_init) + 1):
            scat_height.y_data_init = np.append(scat_height.y_data_init,scat_height.y[-1])
        if (len(scat_height.x) == len(scat.x) + 1):
            scat.x = np.append(scat.x, scat_height.x[-1])               
        
        # calculate sectional voronoi diagram
        vor = sectional_tess.sectional_voronoi(np.transpose(np.array([scat.x,scat.y])),
                                               scat_height.y-scat_height.y_data_init)
        
        # The rest of update_line is based on scipy.spatial.voronoi_plot_2d
        lenridgevert = len(vor.ridge_vertices)
        lin.x = -np.ones(2*lenridgevert,dtype=np.float)
        lin.y = -np.ones(2*lenridgevert,dtype=np.float)
        lin_ext.x = -np.ones(2*lenridgevert,dtype=np.float)
        lin_ext.y = -np.ones(2*lenridgevert,dtype=np.float)
        counter2 = 0
        for isimplex in range(lenridgevert):
            #print vor.ridge_vertices[isimplex]
            simplex = np.asarray(vor.ridge_vertices[isimplex])
            if np.all(simplex >= 0):
                #print simplex
                lin.x[counter2:counter2+2]= vor.vertices[simplex][:,0]
                lin.y[counter2:counter2+2]= vor.vertices[simplex][:,1]
                counter2 += 2
        lin.x = lin.x[:counter2].reshape(counter2//2,2)
        lin.y = lin.y[:counter2].reshape(counter2//2,2)
                
        center = vor.points.mean(axis=0)
        external_scale = np.sqrt(np.std(scat.x)*np.std(scat.y))
        counter2 = 0
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                normt = np.linalg.norm(t)
                if normt > 0.:
                    t /= normt
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[i] + direction*external_scale

                lin_ext.x[counter2:counter2+2]= [vor.vertices[i,0],far_point[0]]
                lin_ext.y[counter2:counter2+2]= [vor.vertices[i,1],far_point[1]]

                counter2 += 2

        lin_ext.x = lin_ext.x[:counter2].reshape(counter2//2,2)
        lin_ext.y = lin_ext.y[:counter2].reshape(counter2//2,2)

        
update_line()
# update line on change of x or y of scatter

scat_height.observe(update_line,names=['y'])

scat.observe(update_line, names=['x'])
scat.observe(update_line, names=['y'])

ax_x = bqplot.Axis(scale=sc_x)
ax_y = bqplot.Axis(scale=sc_y, orientation='vertical')

# change the bleow "with" statements to e.g. disable adding points
with scat_height.hold_sync():
    scat_height.update_on_move = True
    scat_height.update_on_add = True
    scat_height.interactions = {'click': 'add'}
#allow adding generators to 'scat_height' (Fig 1)

with scat.hold_sync():
    scat.update_on_move = True #dynamic update
    scat.update_on_add = True 
    scat.interactions = {'click': 'add'}
#allow adding generators to 'scat' (Fig 2)

bqplot.Figure(marks=[scat_height], axes=[ax_x, ax_y],min_aspect_ratio=1,max_aspect_ratio=1)

bqplot.Figure(marks=[scat, lin, lin_ext], axes=[ax_x, ax_y],min_aspect_ratio=1,max_aspect_ratio=1)



