from fenics import *

get_ipython().magic('matplotlib inline')

def import_mesh(coors, conn):
    '''Construct a mesh from list of nodal coordinates and connectivity table.'''

    dim = coors.shape[1]

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, dim, dim)

    editor.init_vertices(coors.shape[0])  # number of vertices
    editor.init_cells(conn.shape[0])      # number of cells

    for i in range(coors.shape[0]):
        editor.add_vertex(i, coors[i, :])

    for i in range(conn.shape[0]):
        editor.add_cell(i, conn[i, :], )

    editor.close()
    
    return mesh

import numpy as np

# construct a list of nodes of a cube
coors = np.array([[0., 0., 0.],
                  [1., 0., 0.],
                  [1., 1., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.],
                  [1., 0., 1.],
                  [1., 1., 1.],
                  [0., 1., 1.]])

# We define a connectivity table of six tets
tets6a = np.array([[0, 1, 2, 6],
                   [0, 5, 1, 6],
                   [0, 4, 5, 6],
                   [0, 2, 3, 6],
                   [0, 7, 4, 6],
                   [0, 3, 7, 6]], dtype=np.uintp)

mesh = import_mesh(coors, tets6a)

plot(mesh)

import sys

def check_orientation(coords, elems):
    dets = np.zeros(elems.shape[0])

    for i in range(elems.shape[0]):
        nodes = coords[elems[i,:], :]

        dets[i] = np.linalg.det(nodes[[1, 2, 3],:]-nodes[[0, 0, 0],:])

    if min(dets) <= 0.:
        print('Warning: Mimumum volume is not positive.', file=sys.stderr)
        
    return dets

dets = check_orientation(coors, tets6a)
dets

print(mesh.cells())
check_orientation(mesh.coordinates(), mesh.cells())

tets6b = np.array([[0, 1, 3, 4], 
                   [1, 3, 4, 5], 
                   [4, 7, 5, 3],
                   [1, 2, 3, 5], 
                   [2, 3, 5, 7], 
                   [2, 5, 6, 7]], dtype=np.uintp)

mesh = import_mesh(coors, tets6a)
plot(mesh)

dets = check_orientation(coors, tets6b)
dets

tets5 = np.array([[0, 1, 3, 4],
                  [1, 2, 3, 6],
                  [3, 4, 6, 7],
                  [1, 3, 4, 6],
                  [1, 4, 5, 6]], dtype=np.uintp)

mesh = import_mesh(coors, tets6a)
plot(mesh)

dets = check_orientation(coors, tets5)
dets

mesh = UnitCubeMesh(10, 10, 10)
plot(mesh)
mesh.cells().shape

import math
import itertools

def compute_dihedral_angles_tet(nodes):
    '''Compute dihedral angles of a tetrahedron '''
    
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 2, 3], [2, 1, 3]])

    normals = np.zeros([4, 3], dtype=float)
    
    for i in range(4):
        v1 = nodes[faces[i, 1], :] - nodes[faces[i, 0], :]
        v2 = nodes[faces[i, 2], :] - nodes[faces[i, 0], :]
        normals[i, :] = np.cross(v1, v2)
        normals[i, :] /= np.linalg.norm(normals[i, :])

    dihedral_angles = np.zeros(6)
    pairs = np.array(list(itertools.permutations([0, 1, 2, 3], 2)))

    for i in range(6):
        dihedral_angles[i] = math.acos(-np.dot(normals[pairs[i, 0], :],
                             normals[pairs[i, 1], :])) / math.pi * 180.
        
    return dihedral_angles

def compute_dihedral_angles(coors, tets):
    angles = np.zeros([tets.shape[0], 6])

    for i in range(tets.shape[0]):
        angles[i, :] = compute_dihedral_angles_tet(coors[tets[i, :], :])
        
    angles = np.sort(angles)       
    return angles

angles = compute_dihedral_angles(coors, tets6a)

np.set_printoptions(precision=2)
print(angles)

angles = compute_dihedral_angles(coors, tets6b)
print(angles)

angles = compute_dihedral_angles(coors, tets5)
print(angles)

def compute_edge_angles_tet(nodes, edge_angles):
    '''Compute edge angles within a tetrahedron '''
    
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 2, 3], [2, 1, 3]])
    edges = np.array([[0, 1], [1, 2], [2, 0], [0, 1]])

    normals = np.zeros([4, 3], dtype=float)
    
    for i in range(4):
        face = faces[i, :]

        a = [0, 0, 0]

        for j in range(3):
            v1 = nodes[face[edges[j, 1]], :] - nodes[face[edges[j, 0]], :]
            v2 = nodes[face[edges[j+1, 1]], :] - nodes[face[edges[j+1, 0]], :]

            a[j] = (math.acos(-np.dot(v1, v2) / 
                                            np.linalg.norm(v1) / np.linalg.norm(v2)) / 
                                  math.pi * 180.)
            a[j] = round(a[j], 2)
        a.sort()
        
        if a not in edge_angles:
            edge_angles.append(a)

def compute_edge_angles(coors, tets):
    angles = []

    for i in range(tets.shape[0]):
        compute_edge_angles_tet(coors[tets[i, :], :], angles)

    return np.array(angles)

angles = compute_edge_angles(coors, tets6a)
print(angles)

angles = compute_edge_angles(coors, tets6b)
print(angles)

print(compute_edge_angles(coors, tets5))

def compute_longest_edge_tet(nodes):
    '''Compute dihedral angles of a tetrahedron '''

    edges = np.array((list(itertools.combinations([0, 1, 2, 3], 2))))
    lens = np.zeros(6)
    
    for i in range(6):
        v = nodes[edges[i, 1], :] - nodes[edges[i, 0], :]
        lens[i] = np.linalg.norm(v)

    return max(lens)

def compute_longest_edge(coors, tets):
    lens = np.zeros(tets.shape[0])

    for i in range(tets.shape[0]):
        lens[i] = compute_longest_edge_tet(coors[tets[i, :], :])
        
    return max(lens)

print(compute_longest_edge(coors, tets6a))

print(compute_longest_edge(coors, tets6b))

print(compute_longest_edge(coors, tets5))

