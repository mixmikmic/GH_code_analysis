import numpy as np

def parse_name(run_name):
    run_type, cores, parts, steps, time = run_name.split('_')
    if len(cores.split('-')) == 2:
        cores, nodes = cores.split('-')
    else:
        nodes = '1'
    dt = float('{:.3f}'.format(1./float(time))) # round to 3 sig figs
    cores = int(cores)
    nodes = int(nodes)
    if cores > nodes:
        ngpu = 2
    else:
        ngpu = 1
    parts = int(parts)
    steps = int(steps)
    est_hours = 2.0
    est_hours *= steps / 100.
    est_hours *= (parts / 1000.) * np.log(parts / 1000.)
    est_hours /= (cores)**0.5
    hours = int(est_hours % 24)+2
    days = int(est_hours // 24)
    if hours >= 24:
        hours -= 24
        days += 1
    return cores, nodes, ngpu, parts, steps, dt, days, hours

def make_slurm_file(run_name):
    ncores, nnodes, ngpus, nparts, nsteps, dt, days, hours = parse_name(run_name)
    text = '#!/bin/bash\n'
    text += '#SBATCH -p holyseasgpu\n'
    text += '#SBATCH -J {:s} # Job Name\n'.format(run_name)
    text += '#SBATCH -n {:d} # Number of MPI tasks\n'.format(ncores)
    text += '#SBATCH -N {:d} # Fix number of nodes\n'.format(nnodes)
    text += '#SBATCH --gres=gpu:{:d} #Number of GPUs requested per node\n'.format(ngpus)
    text += '#SBATCH --constraint=cuda-7.5 #require CUDA\n'
    text += '#SBATCH -t {:d}-{:02d}:00 # runtime in D-HH:MM\n'.format(days, hours)
    text += '#SBATCH --mem-per-cpu 1536 # memory per MPI task\n'
    text += '#SBATCH -o logs/%x.out\n'
    text += '#SBATCH -e logs/%x.err\n'
    text += '#SBATCH --mail-type=BEGIN,END,FAIL #alert when done\n'
    text += '#SBATCH --mail-user=bcook@cfa.harvard.edu # Email to send to\n\n'
    
    save_every = 10
    if nsteps < 100:
        save_every = 1
    text += 'mpiexec -n $SLURM_NTASKS run_behalf.py --run-name $SLURM_JOB_NAME --clobber --N-parts {:d} --N-steps {:d} --dt {:.3f} --save-every {:d}\n'.format(nparts, nsteps, dt, save_every)
    text += 'RESULT=${PIPESTATUS[0]}\n'
    text += 'sacct -j $SLURM_JOB_ID ----format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State\n'
    text += 'exit $RESULT\n'
    return text

# for name in ['gpuc_1_1000_10_100','gpuc_1_4000_3_100','gpuc_1_16000_3_100', 'gpuc_2_1000_10_100',
#              'gpuc_2-2_1000_10_100','gpuc_4-2_1000_10_100','gpuc_4-4_1000_10_100',
#              'gpuc_8-1_1000_10_100','gpuc_8-2_1000_10_100','gpuc_8-4_1000_10_100','gpuc_8-8_1000_10_100',
#              'gpuc_16-8_1000_10_100','gpuc_32-8_1000_10_100','gpuc_64-8_1000_10_100',
#              'gpuc_128-8_1000_10_100','gpuc_256-8_1000_10_100','gpuc_576-12_1000_10_100',
#              'gpuc_576-12_10000_3_100','gpuc_576-12_100000_3_100',
#              'gpuc_4-2_4000_10_100','gpuc_4-2_16000_3_100','gpuc_8-4_4000_10_100','gpuc_8-4_16000_3_100',
#              'gpuc_8-4_100000_3_100','gpuc_16-8_4000_10_100','gpuc_16-8_16000_3_100',
#              'gpuc_16-8_100000_3_100']:
#     print(name)
#     with open(name + '.slurm', 'w') as f:
#         f.write(make_slurm_file(name))

for name in ['gpuc_576-12_1000_10_100','gpuc_576-12_10000_3_100','gpuc_576-12_4000_3_100',
             'gpuc_576-12_16000_3_100','gpuc_576-12_32000_3_100','gpuc_576-12_64000_3_100',
             'gpuc_4_1000_10_100','gpuc_8-1_1000_10_100','gpuc_8-2_1000_10_100',
             'gpuc_32-8_1000_10_100','gpuc_64-8_1000_10_100','gpuc_128-8_1000_10_100','gpuc_256-8_1000_10_100']:
    print(name)
    with open('../gpuc_scalings_v2/'+name + '.slurm', 'w') as f:
        f.write(make_slurm_file(name))



