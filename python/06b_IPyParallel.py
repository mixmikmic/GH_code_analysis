import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
get_ipython().run_line_magic('matplotlib', 'inline')
import ipyparallel as ipp

rc = ipp.Client(profile='mpi')

view = rc[:]
view.block = True

rc.ids

get_ipython().run_cell_magic('px', '', 'from mpi4py import MPI\ncomm = MPI.COMM_WORLD\nprint("Rank {}/{}".format(comm.rank, comm.size))')

get_ipython().run_cell_magic('px', '', 'from mpi4py import MPI\nimport numpy as np\n\ncomm = MPI.COMM_WORLD\nrank = comm.rank\n\nN = 1e4\nnum_procs = comm.size\n\n#print("Size = {}".format(comm.size))\n\nalph_in = .5\nalph_out = .25\n# Round down to a multiple of num_procs\nN -= N % num_procs\n\nn = int(N / num_procs)\n\ndef in_circle(x,y):\n    return ( (x-.5)**2 + (y-.5)**2 < .5**2)\n\ndef count_pts_in_circle(pts):\n    return np.sum(in_circle(pts[:,0], pts[:,1]))\n\n# Generate points\npts = np.random.rand(n,2)\nm = count_pts_in_circle(pts)\noutside = pts[~in_circle(pts[:,0], pts[:,1]),:]\ninside = pts[in_circle(pts[:,0], pts[:,1]),:]\n\n#print("m={}, inside={}".format(m, len(inside)))\n#print("n-m={}, outside={}".format(n-m, len(outside)))\nprint("Rank {}: m={}".format(rank, m))\nprint()\n\ndel pts\n\n# Send to rank 0\nif rank != 0:\n    comm.send(m, dest=0, tag=1)\n    #comm.Send([pts, MPI.DOUBLE], dest=0, tag=2)\n    comm.Send(outside, dest=0, tag=2)\n    comm.Send(inside, dest=0, tag=3)\n\nelse:\n    import matplotlib.pyplot as plt\n    %matplotlib inline\n    M = m\n    \n    fig = plt.figure(figsize=[8,8])\n    ax = plt.gca()\n    ax.set_aspect(1)\n    plt.xlim(0, 1)\n    plt.ylim(0, 1)\n    \n    ax.plot(inside[:,0],inside[:,1], \'o\', color=\'C0\', alpha=alph_in, label=\'rank0\')\n    ax.plot(outside[:,0],outside[:,1], \'o\', color=\'C0\', alpha=alph_out)\n    \n    for proc in range(1, 4):\n        # Count points\n        m = comm.recv(source=proc, tag=1)\n        M += m\n        \n        # Plot points\n        outside = np.empty([n-m,2])\n        inside = np.empty([m,2])\n        comm.Recv(outside, source=proc, tag=2)\n        comm.Recv(inside, source=proc, tag=3)\n        ax.plot(inside[:,0],inside[:,1], \'o\', color=\'C{}\'.format(proc), alpha=alph_in, label=\'rank{}\'.format(proc))\n        ax.plot(outside[:,0],outside[:,1], \'o\', color=\'C{}\'.format(proc), alpha=alph_out)\n        \n        del inside, outside\n    \n    \n    # Draw circle\n    th = np.linspace(0,2*np.pi, 101)\n    x = .5 * np.cos(th) + .5\n    y = .5 * np.sin(th) + .5\n    ax.plot(x, y, \'k-\')\n    \n    plt.title(\'Monte Carlo P2P $\\pi$ Calculation\')\n    plt.legend(loc=\'upper left\', bbox_to_anchor=[1.01,1])\n    \n    print("N = {}".format(N))\n    \n    print("M = {}".format(M))\n    print("pi ~= {:.5f}".format(4*M/N)) \n    print()')

