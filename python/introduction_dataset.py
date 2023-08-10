get_ipython().run_cell_magic('bash', '', 'cd /data\ndatalad install -r ///workshops/nih-2017/ds000114')

get_ipython().run_cell_magic('bash', '', 'cd /data/ds000114\ndatalad get sub-0[12345]/ses-test/anat\ndatalad get sub-0[12345]/ses-test/func/*fingerfootlips*\ndatalad get derivatives/fmriprep/sub-0[12345]/anat\ndatalad get derivatives/fmriprep/sub-0[12345]/ses-test/func/*fingerfootlips*')

get_ipython().system('tree -L 4 /data/ds000114/')

get_ipython().system('cat /data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-linebisection_events.tsv')

