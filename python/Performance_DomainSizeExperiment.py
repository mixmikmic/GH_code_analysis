get_ipython().run_line_magic('matplotlib', 'inline')

import re
import numpy as np
import pandas as pd
import subprocess
import os
import os.path
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Generate unique filename
test_filename = "domain_size_benchmark_run_0.txt"
file_test = 0
while (os.path.isfile(test_filename)):

    test_filename = "domain_size_benchmark_run_" + str(file_test) + ".txt"
    file_test += 1
    
    
#Use the existing file, do not generate new data
test_filename = "domain_size_benchmark_run_0.txt"
print("Storing data in " + test_filename)

def runBenchmark(filename):
    sim = np.array(["FBL", "CTCS", "KP", "CDKLM"])
    domain_size = np.array([256, 512, 1024, 2048, 3192, 4096, 5192, 6144])
    optimal_block_size =[(32, 8), (32, 4), (32,16), (32,4)] # (block_width, block_height)
    
    with open(test_filename, 'w') as test_file:
        for k in range(len(sim)):
            test_file.write("##########################################################################\n")
            test_file.write("Using simulator " + sim[k] + ".\n")
            test_file.write("##########################################################################\n")
            for i in range(domain_size.shape[0]):

                tic = time.time()

                test_file.write("=========================================\n")
                test_file.write(sim[k] + " [{:02d} x {:02d}]\n".format(domain_size[i], domain_size[i]))
                test_file.write("-----------------------------------------\n")
                cmd = [ "python", "run_benchmark.py",                        "--nx", str(domain_size[i]), "--ny", str(domain_size[i]),                        "--block_width", str(optimal_block_size[k][0]), "--block_height", str(optimal_block_size[k][1]),                        "--simulator", sim[k], "--steps_per_download", "1000"]
                p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output = p.stdout.read()
                test_file.write(output + "\n")
                test_file.write("=========================================\n")
                test_file.write("\n")

                toc = time.time()

                infostr = sim[k] + " [{:02d} x {:02d}] completed in {:.02f} s\n".format(domain_size[i], domain_size[i], (toc-tic))
                test_file.write(infostr)
                test_file.flush()
                os.fsync(test_file)
                print(infostr[:-1])
                    
            test_file.write("\n\n\n")


if not (os.path.isfile(test_filename)):
    runBenchmark(test_filename)
else:
    print("Using existing run in " + test_filename)

def getData(filename):
    # State variables
    simulator = None
    domain_size = None

    data = np.empty((0, 3))

    with open(filename) as origin_file:
        for line in origin_file:

            # Find simulator
            match = re.findall(r'(Using simulator)', line)
            if match:
                simulator = line.split(' ')[2][:-2]

            # Find block size
            match = re.findall(r'(Running with domain size)', line)
            if match:
                domain_size = line.split(' ')[5][1:]

            # Find simulator megacells
            match = re.findall(r'(Maximum megacells)', line)
            if match:
                megacells = float(line.split(' ')[4])
                data = np.append(data, [[simulator, domain_size, megacells]], axis=0)
                
                domain_size = None

    return data

data = getData(test_filename)
print(data)

def setBwStyles(ax):
    from cycler import cycler

    ax.set_prop_cycle( cycler('marker', ['.', 'x', 4, '+', '*', '1']) +
                       cycler('linestyle', ['-.', '--', ':', '-.', '--', ':']) +
                       cycler('markersize', [6, 6, 10, 8, 8, 8]) +
                       cycler('color', ['k', 'k', 'k', 'k', 'k', 'k']) )

simulators = np.unique(data[:,0])

fig = plt.figure()
setBwStyles(fig.gca())

for simulator in simulators:
    print(simulator)
    
    columns = data[:,0] == simulator
    
    domain_sizes = data[columns,1].astype(np.float32)
    megacells = data[columns,2].astype(np.float32)
    
    plt.loglog(domain_sizes*domain_sizes, megacells, label=simulator)

plt.legend(loc=0)

print_domain_sizes = np.array([256, 512, 1024, 2048, 4096])
plt.xticks( print_domain_sizes*print_domain_sizes, map(lambda x: "$" + str(x) + "^2$", print_domain_sizes ) )
plt.xlabel("Domain size")
plt.ylabel("Megacells/s")
plt.savefig(test_filename.replace("txt", "pdf"))





