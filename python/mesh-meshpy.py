get_ipython().magic('matplotlib inline')

import numpy as np

import meshpy
from meshpy.gmsh_reader import read_gmsh
from meshpy.gmsh_reader import (GmshMeshReceiverNumPy, 
                                GmshHexahedralElement,
                                GmshQuadrilateralElement,
                                GmshPoint,
                                GmshIntervalElement)

mr = GmshMeshReceiverNumPy()
read_gmsh(mr, "cylinder.msh")

len(mr.element_types)

elem_type_inds = {}

for i in range(len(mr.element_types)):
    
    e = mr.element_types[i]
    if e in elem_type_inds:
        elem_type_inds[e] += [i]
    else:
        elem_type_inds[e]  = [i]

elem_type_inds.keys()

hex_type  = None
quad_type = None

for t in elem_type_inds.keys():
    
    if isinstance(t, GmshHexahedralElement):
        hex_type  = t
    if isinstance(t, GmshQuadrilateralElement):
        quad_type = t
        
assert hex_type
assert quad_type

hex_inds = elem_type_inds[hex_type]
hex_inds = np.sort(hex_inds)

quad_inds = elem_type_inds[quad_type]
quad_inds = np.sort(quad_inds)

elem_to_node = np.zeros((len(hex_inds),
                         hex_type.node_count()),
                        dtype=np.int)
for i in range(len(hex_inds)):
    ind = hex_inds[i]
    elem_to_node[i,:] = mr.elements[ind]

bndy_face_to_node = np.zeros((len(quad_inds),
                              quad_type.node_count()),
                             dtype=np.int)
bf2n = bndy_face_to_node
for i in range(len(quad_inds)):
    ind = quad_inds[i]
    bf2n[i,:] = mr.elements[ind]

# Nodes array
nodes = np.array(mr.points)

# Switch nodes to lex ordering
inds = hex_type.get_lexicographic_gmsh_node_indices()
elem_to_node = elem_to_node[:,inds]

inds = quad_type.get_lexicographic_gmsh_node_indices()
bndy_face_to_node = bf2n[:,inds]

a = nodes[elem_to_node[3]]
a[np.abs(a)<1e-10]=0
a

hex_type.order

mr.element_markers

