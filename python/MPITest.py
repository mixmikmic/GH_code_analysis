import ipyparallel

# attach to a running cluster
cluster = ipyparallel.Client()#profile='mpi')

print('profile:', cluster.profile)
print('Number of ids:', len(cluster.ids))
print("IDs:", cluster.ids) # Print process id numbers

get_ipython().run_cell_magic('px', '', '\nfrom mpi4py import MPI\n\ncomm = MPI.COMM_WORLD\n\nprint("Hello! I\'m rank %d from %d running in total..." % (comm.rank, comm.size))\n\ncomm.Barrier()   # wait for everybody to synchronize _here_')

get_ipython().run_cell_magic('px', '', '\nfrom mpi4py import MPI\nimport numpy\n\ncomm = MPI.COMM_WORLD\nrank = comm.Get_rank()\n\nprint("Starting")\n# passing MPI datatypes explicitly\nif rank == 0:\n    data = numpy.arange(100, dtype=\'i\')\n    numpy.random.shuffle(data)\n    comm.Send([data, MPI.INT], dest=1, tag=77)\n    print("{0}: sent data to 1: {1}".format(rank, data))\nelif rank == 1:\n    data = numpy.empty(100, dtype=\'i\')\n    comm.Recv([data, MPI.INT], source=0, tag=77)\n    print("{0}: received data from 0: {1}".format(rank, data))\nelse:\n    print("{0}: idle".format(rank))')

get_ipython().run_cell_magic('px', '', '\n#Lets have matplotlib "inline"\n%matplotlib inline\n\n#Python 2.7 compatibility\nfrom __future__ import print_function\n\n#Import packages we need\nimport numpy as np\nfrom matplotlib import animation, rc\nfrom matplotlib import pyplot as plt\n#import mpld3\n\nimport subprocess\nimport os\nimport gc\nimport datetime\n\nimport pycuda.driver as cuda\n\ntry:\n    from StringIO import StringIO\nexcept ImportError:\n    from io import StringIO\n\n#Finally, import our simulator\n#Import our simulator\nfrom SWESimulators import FBL, CTCS, KP07, CDKLM16, PlotHelper, Common, WindStress, IPythonMagic\n#Import initial condition and bathymetry generating functions:\nfrom SWESimulators.BathymetryAndICs import *')

get_ipython().run_cell_magic('px', '', '\n%cuda_context_handler cuda_context')

get_ipython().run_cell_magic('px', '', '\ndef gen_test_data(nx, ny, g, num_ghost_cells):\n    width = 100.0\n    height = 100.0\n    dx = width / float(nx)\n    dy = height / float(ny)\n\n    x_center = dx*nx/2.0\n    y_center = dy*ny/2.0\n    \n    h  = np.zeros((ny+2*num_ghost_cells, nx+2*num_ghost_cells), dtype=np.float32); \n    hu = np.zeros((ny+2*num_ghost_cells, nx+2*num_ghost_cells), dtype=np.float32);\n    hv = np.zeros((ny+2*num_ghost_cells, nx+2*num_ghost_cells), dtype=np.float32);\n    \n    comm = MPI.COMM_WORLD\n\n    #Create a gaussian "dam break" that will not form shocks\n    size = width / 3.0\n    dt = 10**10\n    for j in range(-num_ghost_cells, ny+num_ghost_cells):\n        for i in range(-num_ghost_cells, nx+num_ghost_cells):\n            x = dx*(i+0.5) - x_center\n            y = dy*(j+0.5) - y_center\n            \n            h[j+num_ghost_cells, i+num_ghost_cells] = 0.5 + 0.1*(comm.rank + np.exp(-(x**2/size)))\n            hu[j+num_ghost_cells, i+num_ghost_cells] = 0.1*np.exp(-(x**2/size))\n    \n    max_h_estimate = comm.rank* 0.1 + 0.6\n    max_u_estimate = 0.1*2.0\n    dt = min(dx, dy) / (max_u_estimate + np.sqrt(g*max_h_estimate))\n    \n    return h, hu, hv, dx, dy, dt')

get_ipython().run_cell_magic('px', '', '\n# Set initial conditions common to all simulators\nsim_args = {\n"gpu_ctx": cuda_context,\n"nx": 200, "ny": 100,\n"dx": 200.0, "dy": 200.0,\n"dt": 1,\n"g": 9.81,\n"f": 0.0,\n"r": 0.0\n}')

get_ipython().run_cell_magic('px', '', '\nghosts = [0,0,0,0] # north, east, south, west\ndataShape = (sim_args["ny"] + ghosts[0]+ghosts[2], \n             sim_args["nx"] + ghosts[1]+ghosts[3])\n    \nh0 = np.ones(dataShape, dtype=np.float32) * 60\neta0 = np.zeros(dataShape, dtype=np.float32)\nhu0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32)\nhv0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32)\n    \n#Create bump in to lower left of domain for testing\nif (comm.rank == 0):\n    addUpperCornerBump(eta0, sim_args["nx"], sim_args["ny"], sim_args["dx"], sim_args["dy"], ghosts)\nif (comm.rank == 1):\n    addCentralBump(eta0, sim_args["nx"], sim_args["ny"], sim_args["dx"], sim_args["dy"], ghosts)\nif (comm.rank == 2):\n    addCornerBump(eta0, sim_args["nx"], sim_args["ny"], sim_args["dx"], sim_args["dy"], ghosts)\nif (comm.rank == 3):\n    addLowerLeftBump(eta0, sim_args["nx"], sim_args["ny"], sim_args["dx"], sim_args["dy"], ghosts)\n    \n\nplt.figure(figsize=(12, 8))\nplt.imshow(eta0, origin=\'lower\')\nplt.title("Simulation initial conditions, rank=" + str(comm.rank))')

get_ipython().run_cell_magic('px', '', '\n\n#Initialize simulator\nfbl_args = {"H": h0, "eta0": eta0, "hu0": hu0, "hv0": hv0}\nsim = FBL.FBL(**fbl_args, **sim_args)\n\n#Run a simulation and plot it\nsim.step(300)\neta1, hu1, hv1 = sim.download(interior_domain_only=True)\n\nplt.figure(figsize=(12, 8))\nplt.subplot(1,3,1)\nplt.imshow(eta1, origin=\'lower\')\nplt.subplot(1,3,2)\nplt.imshow(hu1, origin=\'lower\')\nplt.subplot(1,3,3)\nplt.imshow(hv1, origin=\'lower\')\nplt.title("Simulation results, rank=" + str(comm.rank))\nplt.show()')



