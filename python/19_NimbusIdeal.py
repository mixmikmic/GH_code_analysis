from interactiveBoreal import *
import gradutil
from BorealWeights import BorealWeightedProblem

kehys = ReferenceFrame()
nclust = 600
kehys.cluster(nclust=nclust, seedn=6, outdata=kehys.x_stack)
data = kehys.centers
nvar = len(kehys.x_norm)
weights_norm = kehys.weights/nvar
ideal = kehys.ideal
nadir = kehys.nadir

solver_name = 'cplex'

nimbus_ref = np.array([[kehys.ideal[0], kehys.nadir[1], kehys.nadir[2], kehys.nadir[3]],
                       [kehys.nadir[0], kehys.ideal[1], kehys.nadir[2], kehys.nadir[3]],
                       [kehys.nadir[0], kehys.nadir[1], kehys.ideal[2], kehys.nadir[3]],
                       [kehys.nadir[0], kehys.nadir[1], kehys.nadir[2], kehys.ideal[3]]])
stay = np.array([], dtype=int)
detoriate = np.array([], dtype=int)
nimbus_res_cluster = []
nimbus_res_orig = []
nimbus_problems = []
ide = kehys.normalize_ref(kehys.ideal)
nad = kehys.normalize_ref(kehys.nadir)
for i in range(len(nimbus_ref)):
    minmax = np.array([i], dtype=int)
    ref = kehys.normalize_ref(nimbus_ref[i])
    nimbus_problems.append(NIMBUS(ide, nad, ref, kehys.centers, 
                                  minmax, stay, detoriate, np.array([0,0,0,0]), 
                                  weights=kehys.weights, nvar=nvar))
    nimbus_solver = Solver(nimbus_problems[i].model, solver=solver_name)
    res = nimbus_solver.solve() 
    nimbus_res_cluster.append([gradutil.model_to_real_values(kehys.x_stack[:,:,j], 
                                                             nimbus_problems[i].model, 
                                                             kehys.xtoc) 
                               for j in range(len(nimbus_ref))])
    nimbus_res_orig.append([gradutil.cluster_to_value(kehys.x_stack[:,:,j], 
                                                      gradutil.res_to_list(nimbus_problems[i].model), 
                                                      kehys.weights)
                            for j in range(len(nimbus_ref))])

for k in nimbus_res_cluster:
     print(['{:12.2f}'.format(value) for value in k])

for k in nimbus_res_orig:
    print(['{:12.2f}'.format(value) for value in k])

nimbus_ideal = np.max(nimbus_res_cluster, axis=0)
print(['{:12.2f}'.format(value) for value in nimbus_ideal])

nimbus_nadir = np.min(nimbus_res_cluster, axis=0)
print(['{:12.2f}'.format(value) for value in nimbus_nadir])

print(['{:12.2f}'.format(value) for value in kehys.ideal])
print(['{:12.2f}'.format(value) for value in kehys.nadir])

