get_ipython().magic('run EUVIP_1_defaults.ipynb')

get_ipython().magic('cd -q ../test/')

get_ipython().run_cell_magic('writefile', '../test/experiment_sparseness.py', '# -*- coding: utf8 -*-\nfrom __future__ import division, print_function\n"""\n\nExploring the sparseness of coefficients in the SparseEdges framework.\n\n"""\nimport sys\nexperiment = sys.argv[1]\nparameter_file = sys.argv[2]\nname_database = sys.argv[3]\nN_image = int(sys.argv[4])\nprint(\'N_image = \', N_image)\nN = int(sys.argv[5])\ndo_linear = (sys.argv[6] == \'True\')\n\nimport numpy as np\nfrom SparseEdges import SparseEdges\nmps = []\nfor name_database in [name_database]:\n    mp = SparseEdges(parameter_file)\n    mp.pe.datapath = \'database/\'\n    mp.pe.N_image = N_image\n    mp.pe.do_mask = True\n    mp.pe.N = N\n    mp.init()\n    # normal experiment\n    imageslist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)\n    mps.append(mp)\n    # control experiment\n    if do_linear:\n        mp.pe.MP_alpha = np.inf\n        mp.init()\n        imageslist, edgeslist, RMSE = mp.process(exp=experiment + \'_linear\', name_database=name_database)\n        mps.append(mp)')

experiment_folder = experiment = 'EUVIP-sparseness'

cluster = True
cluster = False

if cluster:
    try:
        from INT_cluster import Frioul
        k = Frioul(experiment_folder, N_jobs = 64)
    except Exception:
        cluster = False
else:
    def run_cmd(cmd, doit=True):
        import subprocess
        print ('⚡︎ Running ⚡︎ ', cmd)
        if doit:
            stdout = subprocess.check_output([cmd], shell=True)
            return stdout.decode()#.splitlines()


do_update = True
do_update = False

do_cleanup = True
do_cleanup = False

do_run = True
do_run = False

# update
if cluster and do_update:
    k.do_update()
    
# clean-up
if cluster and do_cleanup:
    for cmd in [
        #"rm -fr results data_cache ",
        #"find . -name *sparselets* -exec rm -fr {} \\;",
        "find . -name *lock* -exec rm -fr {} \\;",
        #"touch frioul; rm frioul* ",
        ]:
        print(k.run_on_cluster(cmd))

# preparing
if do_run:
    # RUNNING
    if cluster: k.push_to_cluster(source="{../test/results,../test/data_cache,../test/experiment_sparseness.py,../database}")

    args = 'experiment_sparseness.py {experiment} {parameter_file} {name_database} {N_image} {N} {do_linear} '.format(
            experiment=experiment, parameter_file=parameter_file, 
            name_database=name_database, N_image=N_image, N=N, do_linear=do_linear)

    if cluster:
        fullcmd = 'ipython {args}'.format(args=args)
        for cmd in [
            "frioul_batch  -M {N_jobs} '{fullcmd}' ".format(N_jobs=k.N_jobs, fullcmd=fullcmd), 
            "frioul_list_jobs -v |grep job_array_id |uniq -c",
                    ]:
            print(k.run_on_cluster(cmd))
    else:
        fullcmd = 'ipython3 {args}'.format(args=args)
        run_cmd (fullcmd)

# GETTING the data
import time, os
if cluster:
    while True:    
        print(k.pull_from_cluster())
        print(k.run_on_cluster("tail -n 10 {}".format(os.path.join(k.PATH, 'debug.log'))))
        print(k.run_on_cluster("frioul_list_jobs -v |grep job_array_id |uniq -c"))
        locks = k.run_cmd ("find . -name *lock -exec ls -l {} \;")
        print(locks)
        if len(locks) == 0: break
        time.sleep(100)    

get_ipython().magic('run experiment_sparseness.py EUVIP-sparseness https://raw.githubusercontent.com/bicv/SparseEdges/master/default_param.py serre07_distractors 100 4096 True')

imageslist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)

fig, [A, B] = plt.subplots(1, 2, figsize=(fig_width, fig_width/1.618), subplot_kw={'axisbg':'w'})
A.set_color_cycle(np.array([[1., 0., 0.]]))
imagelist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)
RMSE /= RMSE[:, 0][:, np.newaxis]
#print( RMSE.shape, edgeslist.shape)
value = edgeslist[4, ...]
#value /= value[0, :][np.newaxis, :]
value /= RMSE[:, 0][np.newaxis, :]

B.semilogx( value, alpha=.7, lw=.1)

A.semilogx( RMSE.T, alpha=.7, lw=.1)
for ax in [A,B]:
    ax.set_xlabel('rank of coefficient')
    ax.axis('tight')
_ = A.set_ylabel('RMSE')
_ = B.set_ylabel('coefficient')


#plt.locator_params(axis = 'x', nbins = 5)
#plt.locator_params(axis = 'y', nbins = 5)
mp.savefig(fig, experiment + '_raw', figpath = '../docs/');

#fig = plt.figure(figsize=(fig_width, fig_width/1.618))
fig = plt.figure(figsize=(fig_width, fig_width/1.5))

if do_linear:
    fig, ax, inset = mp.plot(mps=[mp]*2, experiments=[experiment, experiment + '_linear'], databases=[name_database]*2, fig=fig, 
                  color=[0., 0., 1.], scale=False, labels=['MP', 'lin'])
else:
    fig, ax, inset = mp.plot(mps=[mp], experiments=[experiment], databases=[name_database], fig=fig, 
                  color=[0., 0., 1.], scale=False, labels=['MP'])
    
ax.set_yticks([0, 1.])            
inset.set_yticks([])            
ax.set_xticks([])
ax.set_xticks([0, 2048, 4096])            
inset.set_xticks([0, 2048, 4096])            

mp.savefig(fig, experiment + '_raw_inset', figpath = '../docs/');

get_ipython().magic('cd -q ../notebooks/')

