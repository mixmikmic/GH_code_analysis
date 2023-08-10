from subprocess import getoutput as sgo
import pygimli.meshtools.polytools as plc
import numpy as np

def print2file(string,file='figs/3d_mesh.geo'):
    sgo('echo "'+string+'" >> '+file)
    

radius_inner = 100
radius_outer = 140
world_height = 400
world_width = 200
pipe_length = 300
segments = 40
characteristicLengthMin = 25

assert radius_inner < radius_outer
assert radius_outer < world_height
assert radius_outer < world_width

c_inner = plc.createCircle([0, 0], radius=radius_inner, segments=segments)
c_outer = plc.createCircle([0, 0], radius=radius_outer, segments=segments)

get_ipython().system('rm figs/3d_mesh.geo')
print2file('lc1 = 18;')
print2file('lc2 = 118;')

pointIdx = 1
points_outer = []
for node in c_outer.nodes():
    print2file('Point('+str(pointIdx)+') ={'+str(node.x())+', '+                                        str(node.y())+', '+                                        str(node.z())+', lc1};')
    points_outer.append(pointIdx)
    pointIdx += 1
print2file(' ')

points_inner = []
for node in c_inner.nodes():
    print2file('Point('+str(pointIdx)+') ={'+str(node.x())+', '+                                        str(node.y())+', '+                                        str(node.z())+', lc1};')
    points_inner.append(pointIdx)
    pointIdx += 1    

#Create line entities
lineIdx = 1
for i,pointIdx in enumerate(points_outer):
    if i == 0:
        startIdx = pointIdx
    if i < len(points_outer)-1:
        print2file('Line('+str(lineIdx)+') ={'+str(pointIdx)+', '+str(pointIdx+1)+'};')
        lineIdx += 1
    elif i == len(points_outer)-1:
        print2file('Line('+str(lineIdx)+') ={'+str(pointIdx)+', '+str(startIdx)+'};')
        lineIdx += 1
print2file(' ')        
        
for i,pointIdx in enumerate(points_inner):
    if i == 0:
        startIdx = pointIdx
    if i < len(points_inner)-1:
        print2file('Line('+str(lineIdx)+') ={'+str(pointIdx)+', '+str(pointIdx+1)+'};')
        lineIdx += 1
    elif i == len(points_outer)-1:
        print2file('Line('+str(lineIdx)+') ={'+str(pointIdx)+', '+str(startIdx)+'};')
        lineIdx += 1
        
#Create line loop entities
lineLoopOuterIdx = lineIdx
print2file('Line Loop('+str(lineLoopOuterIdx)+') = {'+str(points_outer)[1:-1]+'};')

lineIdx += 1
lineLoopInnerIdx = lineIdx
print2file('Line Loop('+str(lineLoopInnerIdx)+') = {'+str(points_inner)[1:-1]+'};')

lineIdx += 1
print2file('Plane Surface('+str(lineIdx)+') = {'+str(lineLoopOuterIdx)+', '+str(lineLoopInnerIdx)+'};')

print2file(' ')    

print2file('Extrude {0, 0, '+str(pipe_length)+'} {')
print2file('  Surface{'+str(lineIdx)+'};')
print2file('}')



#Outer world
worldEntityIdx1 = pointIdx + lineIdx + 10**(np.ceil(np.log10(pointIdx+lineIdx)))
worldEntityIdx2 = worldEntityIdx1 + 1
worldEntityIdx3 = worldEntityIdx1 + 2
worldEntityIdx4 = worldEntityIdx1 + 3

print2file('Point('+str(worldEntityIdx1)+') ={'+str(-world_width)+', '+                                                str(world_height)+', '+                                                str(0)+', lc2};')
print2file('Point('+str(worldEntityIdx2)+') ={'+str(world_width)+', '+                                                str(world_height)+', '+                                                str(0)+', lc2};')
print2file('Point('+str(worldEntityIdx3)+') ={'+str(world_width)+', '+                                                str(-world_height)+', '+                                                str(0)+', lc2};')
print2file('Point('+str(worldEntityIdx4)+') ={'+str(-world_width)+', '+                                                str(-world_height)+', '+                                                str(0)+', lc2};')

worldLineIdx5 = worldEntityIdx1 + 4
worldLineIdx6 = worldEntityIdx1 + 5
worldLineIdx7 = worldEntityIdx1 + 6
worldLineIdx8 = worldEntityIdx1 + 7
print2file(' ') 

print2file('Line('+str(worldLineIdx5)+') ={'+str(worldEntityIdx1)+', '+str(worldEntityIdx2)+'};')
print2file('Line('+str(worldLineIdx6)+') ={'+str(worldEntityIdx2)+', '+str(worldEntityIdx3)+'};')
print2file('Line('+str(worldLineIdx7)+') ={'+str(worldEntityIdx3)+', '+str(worldEntityIdx4)+'};')
print2file('Line('+str(worldLineIdx8)+') ={'+str(worldEntityIdx4)+', '+str(worldEntityIdx1)+'};')
print2file(' ') 

worldLineLoopIdx9 = worldEntityIdx1 + 8
print2file('Line Loop('+str(worldLineLoopIdx9)+           ') = {'+str(worldLineIdx5)+', '+           str(worldLineIdx6)+', '+           str(worldLineIdx7)+', '+           str(worldLineIdx8)+'};')

worldPlaneSurfaceIdx10 = worldEntityIdx1 + 9
print2file('Plane Surface('+str(worldPlaneSurfaceIdx10)+') = {'+str(lineLoopOuterIdx)+', '+str(worldLineLoopIdx9)+'};')

print2file(' ')    

print2file('Extrude {0, 0, '+str(pipe_length)+'} {')
print2file('  Surface{'+str(worldPlaneSurfaceIdx10)+'};')
print2file('}')

#print2file('Mesh.CharacteristicLengthMin = '+str(characteristicLengthMin)+';')

# http://gmsh.info/doc/texinfo/gmsh.html#Mesh-options-list
from pygimli.meshtools import readGmsh
import subprocess

subprocess.call(["gmsh", "-3", "-o", "figs/3d_mesh.msh", "figs/3d_mesh.geo"])
mesh = readGmsh('figs/3d_mesh.msh', verbose=False)

for cell in mesh.cells():
    distanceFromOrigin = np.sqrt((cell.center().x())**2+(cell.center().y())**2)
    if distanceFromOrigin > radius_outer:
        cell.setMarker(2)
        
mesh.save('figs/3d_mesh.bms')
mesh.exportVTK('figs/3d_mesh')
print(mesh)

get_ipython().system('paraview --data=figs/3d_mesh.vtk')

#Set boundary conditions
import numpy as np
import pygimli as pg

outer_boundaries = 0
for bound in mesh.boundaries():
    try:
        bound.leftCell().id()
        existLeftCell = True    
    except:
        existLeftCell = False

    try:
        bound.rightCell().id()
        existRightCell = True    
    except:
        existRightCell = False

    if np.array([existLeftCell,existRightCell]).all() == False:
        bound.setMarker(pg.MARKER_BOUND_HOMOGEN_NEUMANN)
        outer_boundaries += 1

print(outer_boundaries)

#Specify electrode nodes
#240, 641
for i,node in enumerate(mesh.nodes()):
    if i == 240:
        elec1_x, elec1_y, elec1_z = node.x(),node.y(),node.z()
        print(elec1_x, elec1_y, elec1_z)
    if i == 141:
        elec2_x, elec2_y, elec2_z = node.x(),node.y(),node.z()
        print(elec2_x, elec2_y, elec2_z)        
    if i == 842:
        elec3_x, elec3_y, elec3_z = node.x(),node.y(),node.z()
        print(elec3_x, elec3_y, elec3_z)        
    if i == 843:
        elec4_x, elec4_y, elec4_z = node.x(),node.y(),node.z() 
        print(elec4_x, elec4_y, elec4_z)        

#Useful if dcmod is applied
get_ipython().system('echo "4# Number of electrodes" > figs/config.shm')
get_ipython().system('echo "#x y z" >> figs/config.shm')
get_ipython().system('echo "{elec1_x} {elec1_y} {elec1_z}" >> figs/config.shm')
get_ipython().system('echo "{elec2_x} {elec2_y} {elec2_z}" >> figs/config.shm')
get_ipython().system('echo "{elec3_x} {elec3_y} {elec3_z}" >> figs/config.shm')
get_ipython().system('echo "{elec4_x} {elec4_y} {elec4_z}" >> figs/config.shm')
get_ipython().system('echo "4# Number of data" >> figs/config.shm')
get_ipython().system('echo "#a b m n" >> figs/config.shm')
get_ipython().system('echo "1 2 3 4" >> figs/config.shm')
get_ipython().system('echo "2 1 3 4" >> figs/config.shm')
get_ipython().system('echo "1 4 3 2" >> figs/config.shm')
get_ipython().system('echo "2 3 1 4" >> figs/config.shm')

get_ipython().system('cat figs/config.shm')

#Set electrode marker
for node in mesh.nodes():
    if (node.x() == elec1_x) and         (node.y() == elec1_y) and         (node.z() == elec1_z):
            print(node.id())
            node.setMarker(99)  
    if (node.x() == elec2_x) and         (node.y() == elec2_y) and         (node.z() == elec2_z):
            print(node.id())        
            node.setMarker(99)
    if (node.x() == elec3_x) and         (node.y() == elec3_y) and         (node.z() == elec3_z):
            print(node.id())        
            node.setMarker(99)
    if (node.x() == elec4_x) and         (node.y() == elec4_y) and         (node.z() == elec4_z):
            print(node.id())        
            node.setMarker(99)
    
mesh.save('figs/3d_mesh')

#Explicit definition of boundary conditions
from pygimli.solver import solve

def mixedBC(boundary, userData):
    sourcePos = userData['sourcePos']
    k = userData['k']
    r1 = boundary.center() - sourcePos
    # Mirror on surface at depth=0
    r2 = boundary.center() - pg.RVector3(1.0, -1.0, 1.0) * sourcePos
    r1A = r1.abs()
    r2A = r2.abs()

    n = boundary.norm()

    if r1A > 1e-12 and r2A > 1e-12:
        return k * ((r1.dot(n)) / r1A * pg.besselK1(r1A * k) +
                    (r2.dot(n)) / r2A * pg.besselK1(r2A * k)) / \
            (pg.besselK0(r1A * k) + pg.besselK0(r2A * k))
    else:
        return 0.
    
def pointSource(cell, f, userData):
    sourcePos = userData['sourcePos']

    if cell.shape().isInside(sourcePos):
        f.setVal(cell.N(cell.shape().rst(sourcePos)), cell.ids())


sourcePosA = [elec1_x, elec1_y, elec1_z]
sourcePosB = [elec2_x, elec2_y, elec2_z]

k = 1e-3
sigma = np.zeros(mesh.cellCount())
for i,cell in enumerate(mesh.cells()):
    sigma[i] = 1000 * (cell.marker()+.1)


u1 = solve(mesh, a=sigma, b=sigma * k*k, f=pointSource,
          duB=[[-1, mixedBC]],
          userData={'sourcePos': sourcePosA, 'k': k},
          verbose=True)

#For the sake of simplicity, mixedBC is used for all boundary facets
u2 = solve(mesh, a=sigma, b=sigma * k*k, f=pointSource,
          duB=[[-1, mixedBC]],
          userData={'sourcePos': sourcePosB, 'k': k},
          verbose=True)

u = u1 - u2

#Write electric field into vtk
try:
    get_ipython().system('rm figs/3d_mesh.vtk')
except:
    pass

mesh.save('figs/3d_mesh')
mesh.exportVTK('figs/3d_mesh')
get_ipython().system("echo 'SCALARS valuesC double 1' >> figs/3d_mesh.vtk")
get_ipython().system("echo 'LOOKUP_TABLE default' >> figs/3d_mesh.vtk")
get_ipython().system("echo {str(list(u)).replace('[','').replace(']','').replace('\\n','').replace(',','')} >> figs/3d_mesh.vtk")

get_ipython().system('paraview --data=figs/3d_mesh.vtk')

print(np.unique(np.sort(sigma)))



