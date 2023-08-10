import pandas as pd
import rmats_inclevel_analysis as r

clip_manifest = pd.read_table('/home/bay001/projects/maps_20160420/permanent_data/ALLDATASETS_submittedonly.txt')
annotation_dir = '/projects/ps-yeolab3/bay001/maps/current_annotations/se/'
hepg2_rnaseq_manifest = '/home/bay001/projects/maps_20160420/permanent_data/RNASeq_final_exp_list_HepG2.csv'
k562_rnaseq_manifest = '/home/bay001/projects/maps_20160420/permanent_data/RNASeq_final_exp_list_K562.csv'
rnaseq_manifests = {'HepG2':hepg2_rnaseq_manifest, 'K562':k562_rnaseq_manifest}
pos_suffix = '.positive.nr.txt'
neg_suffix = '.negative.nr.txt'

out_file = '/home/bay001/projects/codebase/data/test_num_differential_events.txt'

df = r.run_num_differential_events(
    clip_manifest_df=clip_manifest, 
    rnaseq_manifests_dict=rnaseq_manifests, 
    annotation_dir=annotation_dir, 
    pos_suffix=pos_suffix, 
    neg_suffix=neg_suffix, 
)

f = '/projects/ps-yeolab3/bay001/maps/current_annotations/se/RBFOX2-BGHLV26-HepG2-SE.MATS.JunctionCountOnly.txt'
out_file = '/home/bay001/projects/codebase/data/test_num_differential_events_violin.png'

violin_df = r.make_df_for_violin_plot(f) # to just get the underlying dataframe
r.make_violin_plot_single_rbp(f, out_file)

f = '/projects/ps-yeolab3/bay001/maps/current_annotations/se/RBFOX2-BGHLV26-HepG2-SE.MATS.JunctionCountOnly.txt'
df = pd.read_table(f)
r.get_hist(df)



