get_ipython().magic('run util_notebook.py')
get_ipython().magic('run -i config.py')
get_ipython().magic('run -i geometry.py')
get_ipython().magic('run -i matching.py')

DIR_OUT_EVAL

# Load the evaluation measurement files

# evaluated on syntetic
vs_all_esyn = merge_accs_from_dir(pp(DIR_OUT_EVAL, 'eval_synth_all_long'))

# evaluated on architectural
vs_all_earch = merge_accs_from_dir(pp(DIR_OUT_EVAL, 'eval_arch_all_long'))

# evaluated on 7scenes
vs_all_e7sc = acc_merge_list([
	merge_accs_from_dir(pp(DIR_OUT_EVAL, 'eval_7sc_arch-7sc_short')),
	merge_accs_from_dir(pp(DIR_OUT_EVAL, 'eval_7sc_syn-7sc_short')),
])

# Number of detected points

plot_acc_gt(vs_all_earch, shard=10,
	save=pp(DIR_OUT_FIGURES, 'gt', 'gt_earch.pdf')
)

plot_acc_gt(vs_all_esyn, shard=10,
	save=pp(DIR_OUT_FIGURES, 'gt', 'gt_esyn.pdf')
)

plot_acc_gt(vs_all_e7sc, shard=10,
	save=pp(DIR_OUT_FIGURES, 'gt', 'gt_e7sc.pdf')
)

# Comparison between training datasets

plot_acc(
	vs_all_esyn, 
	#eval_ds = 'Synthetic',
	shard = 10,
	desc_ids = [
		#('flat', 'sift', 'SIFT'),
		('flat', 'net_arch_int', 'Outdoor network'),
		('flat', 'net_syn_int', 'Synthetic network'),
		('flat', 'net_7sc_int', 'Indoor network'),
	],
	save=pp(DIR_OUT_FIGURES, 'dset_comp', 'dset_accuracy_esyn.pdf'),
) 

plot_acc(
	#vs_all_earch,
	vs_all_earch,
	#eval_ds = 'Architectural',
	shard = 10,
	desc_ids = [
		#('flat', 'sift', 'SIFT'),
		('flat', 'net_arch_int', 'Outdoor network'),
		('flat', 'net_syn_int', 'Synthetic network'),
		('flat', 'net_7sc_int', 'Indoor network'),
		
	],
	save=pp(DIR_OUT_FIGURES, 'dset_comp', 'dset_accuracy_earch.pdf'),
)

plot_acc(
	vs_all_e7sc, 
	#eval_ds = '7 Scenes',
	shard = 10,
	desc_ids = [
		('flat', 'sift', 'SIFT'),
		('flat', 'net_arch_int', 'Outdoor network'),
		('flat', 'net__syn_int', 'Synthetic network'),
		('flat', 'net_7sc_int', 'Indoor network'),
		
	]
)

# Comparison between flat / depth / normals networks

plot_acc(
	vs_all_earch, 
	shard = 10,
	desc_ids = [
		('flat', 'sift', 'SIFT'),
		('flat', 'net_arch_int', 'Standard network'),
		('flat', 'net_arch_depth', 'Depth network'),
		('flat', 'net_arch_norm', 'Normals network'),
	],
	save = pp(DIR_OUT_FIGURES, 'depth', 'std_depthnorm_arch_earch.pdf')
)

plot_acc(
	vs_all_earch, 
	shard = 10,
	desc_ids = [
		('flat', 'sift', 'SIFT'),
		('flat', 'net_7sc_int', 'Standard network'),
		('flat', 'net_7sc_depth', 'Depth network'),
		('flat', 'net_7sc_norm', 'Normals network'),
	],
	save = pp(DIR_OUT_FIGURES, 'depth', 'std_depthnorm_7sc_earch.pdf')
)

plot_acc(
	vs_all_esyn, 
	shard = 10,
	desc_ids = [
		('flat', 'sift', 'SIFT'),
		('flat', 'net_arch_int', 'Standard network'),
		('flat', 'net_arch_depth', 'Depth network'),
		('flat', 'net_arch_norm', 'Normals network'),
	],
	save = pp(DIR_OUT_FIGURES, 'depth', 'std_depthnorm_arch_esyn.pdf')
)

plot_acc(
	vs_all_esyn, 
	shard = 10,
	desc_ids = [
		('flat', 'sift', 'SIFT'),
		('flat', 'net_syn_int', 'Standard network'),
		('flat', 'net_syn_depth', 'Depth network'),
		('flat', 'net_syn_norm', 'Normals network'),
	],
	save = pp(DIR_OUT_FIGURES, 'depth', 'std_depthnorm_syn_esyn.pdf')
) 

# Unwarp before or after detection

plot_acc(
	vs_all_esyn, 
	#eval_ds = 'Synthetic',
	shard = 10,
	desc_ids = [
		('flat', 'sift', 'original'),
		('unwarp', 'sift', 'unwarped-after-detection'),
		('unwarp_det', 'sift', 'unwarped-after-detection'),
	],
	save = pp(DIR_OUT_FIGURES, 'unw_comp', 'unw_comp_esyn.pdf'),
) 

plot_acc(
	vs_all_earch,
	#eval_ds = 'Architectural',
	shard = 10,
	desc_ids = [
		('flat', 'sift', 'original'),
		('unwarp', 'sift', 'unwarped-after-detection'),
		('unwarp_det', 'sift', 'unwarped-after-detection'),
	],
	save = pp(DIR_OUT_FIGURES, 'unw_comp', 'unw_comp_earch.pdf')
)







