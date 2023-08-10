#NAME: Crystal Growth
#DESCRIPTION: Statistical mechanics model of the growth of a crystal.

from vpython import *
import numpy as np

cube_size = 10
x_offset, y_offset, z_offset = 0, 0, 0

surface = [np.array((1,0,0), dtype = np.int), 
           np.array((-1,0,0), dtype = np.int), 
           np.array((0,1,0), dtype = np.int), 
           np.array((0,-1,0), dtype = np.int), 
           np.array((0,0,1), dtype = np.int), 
           np.array((0,0,-1), dtype = np.int)]

L = 1000
sites = np.zeros((L, L, L), dtype = np.int)
site_list = [np.array((0,0,0), dtype = np.int)]    

sites[0,0,0] = 6
sites[1,0,0] = 1
sites[-1,0,0] = 1
sites[0,1,0] = 1
sites[0,-1,0] = 1
sites[0,0,1] = 1
sites[0,0,-1] = 1

accept_probs = [0.01, 0.01, 1, 1, 1, 1]

first_neighbours = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0) ,(0,0,1), (0,0,-1)]

second_neighbours = []
for a in (1,-1):
    for b in (1,-1):
        for c in ((a,b,0), (a,0,b), (0,a,b)):
            second_neighbours.append(c)

fmin = 0 # This is the number of faces you need to be added to surface

for step in range(100000):
    surface_index = np.random.randint(len(surface))
    surface_box = surface[surface_index]  # Choose point on the outer surface at random
    occupied_faces = sites[tuple(surface_box)] # The number of occupied faces
    
    # Check that there are occupied second neighbours. If there are not, don't fill
    if step > 1000 & any([sites[tuple(surface_box + vec)] for vec in second_neighbours]):
    
        if np.random.rand() < accept_probs[occupied_faces - 1]:         # We are going to add the cube

            site_list.append(surface_box)
            # box(pos=vector(x_offset + surface_box[0] * cube_size, y_offset + surface_box[1] * cube_size,z_offset + surface_box[2] * cube_size),size = vector(cube_size,cube_size,cube_size)) # Draw the box

            # remove site from outer surface
            del(surface[surface_index])

            #increment surrounding sites and add to surface if new

            if step == 1000:
                fmin = 2 # After some time, we only add sites to the surface if they have fmin + 1 neighbours
            
            for neighbour in first_neighbours:
                
                if sites[tuple(surface_box + neighbour)] == fmin:
                    surface.append(surface_box + neighbour)
                sites[tuple(surface_box + neighbour)] += 1 

            
        
            


scene = canvas()
scene.forward = vector(-1, -1, -1)

for site_index in site_list:
    if sites[tuple(site_index)] < 6:
        box(pos=vector(x_offset + site_index[0] * cube_size, y_offset + site_index[1] * cube_size,z_offset + site_index[2] * cube_size),size = vector(cube_size,cube_size,cube_size), color = color.white)

