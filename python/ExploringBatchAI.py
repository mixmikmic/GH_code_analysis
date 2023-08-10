get_ipython().run_line_magic('run', 'setup_bait.py')

setup_cluster()

wait_for_cluster()

print_cluster_list()

submit_all()

print_jobs_summary()

wait_for_job('run_cntk')

download_files('run_cntk', 'notebooks')

delete_all_jobs()

delete_cluster()

print_status()

