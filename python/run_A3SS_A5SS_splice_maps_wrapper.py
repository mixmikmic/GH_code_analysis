import pandas as pd
import os
import json
import yaml
import glob
# import rethinkdb as r
from collections import defaultdict
from qtools import Submitter
from encode import manifest_helpers as m

from tqdm import tnrange, tqdm_notebook
pd.set_option("display.max_colwidth", 10000)



current_date = '12-5-2017'
clip_manifest = '/home/bay001/projects/maps_20160420/permanent_data/ALLDATASETS_submittedonly.txt'
hepg2_rnaseq_manifest = '/home/bay001/projects/maps_20160420/permanent_data/RNASeq_final_exp_list_HepG2.csv'
k562_rnaseq_manifest = '/home/bay001/projects/maps_20160420/permanent_data/RNASeq_final_exp_list_K562.csv'
rnaseq_manifests = {'HepG2':hepg2_rnaseq_manifest, 'K562':k562_rnaseq_manifest}
density_runner = '/home/bay001/projects/codebase/rbp-maps/maps/plot_density.py'

a5ss_k562_all = 'K562-all-native-a5ss-events'
a5ss_k562_basic = 'K562-shorter-isoform-in-majority-of-controls'
a5ss_k562_center = 'K562-mixed-psi-isoform-in-majority-of-controls'
a5ss_k562_extension = 'K562-longer-isoform-in-majority-of-controls'

a3ss_k562_all = 'K562-all-native-a3ss-events'
a3ss_k562_basic = 'K562-shorter-isoform-in-majority-of-controls'
a3ss_k562_center = 'K562-mixed-psi-isoform-in-majority-of-controls'
a3ss_k562_extension = 'K562-longer-isoform-in-majority-of-controls'

a5ss_hepg2_all = 'HepG2-all-native-a5ss-events'
a5ss_hepg2_basic = 'HepG2-shorter-isoform-in-majority-of-controls'
a5ss_hepg2_center = 'HepG2-mixed-psi-isoform-in-majority-of-controls'
a5ss_hepg2_extension = 'HepG2-longer-isoform-in-majority-of-controls'

a3ss_hepg2_all = 'HepG2-all-native-a3ss-events'
a3ss_hepg2_basic = 'HepG2-shorter-isoform-in-majority-of-controls'
a3ss_hepg2_center = 'HepG2-mixed-psi-isoform-in-majority-of-controls'
a3ss_hepg2_extension = 'HepG2-longer-isoform-in-majority-of-controls'

clip_df = pd.read_table(clip_manifest)

events = {
    'a3ss':'/projects/ps-yeolab3/bay001/maps/current_annotations/a3ss_renamed/',
    'a5ss':'/projects/ps-yeolab3/bay001/maps/current_annotations/a5ss_renamed/',
}

img_extensions = ['png']
out_base = '/projects/ps-yeolab3/bay001/maps/current/'
# out_base = '/home/bay001/projects/maps_20160420/analysis/'

pos_splicing_suffix = '-longer-isoform-included-upon-knockdown'
neg_splicing_suffix = '-shorter-isoform-included-upon-knockdown'


for event, annotation_dir in events.iteritems(): # for each annotation
    for img_extension in img_extensions: # for each image extension
        no_rnaseq = [] # uIDs for which we don't have rna seq expt ids for
        no_rnaseq_yet = [] # uIDs for which we have an expt id, but haven't downloaded the data yet
        cmds = []
        output_dir = os.path.join(out_base, '{}'.format(event))
        for uid in clip_df['uID']:
            r1, r2, i, rbp, cell = m.get_clip_file_from_uid(clip_df, uid)

            if cell == 'K562':
                if event == 'a3ss':
                    background_all = os.path.join(annotation_dir, a3ss_k562_all)
                    background_basic = os.path.join(annotation_dir, a3ss_k562_basic)
                    background_center = os.path.join(annotation_dir, a3ss_k562_center)
                    background_extension = os.path.join(annotation_dir, a3ss_k562_extension)
                elif event == 'a5ss':
                    background_all = os.path.join(annotation_dir, a5ss_k562_all)
                    background_basic = os.path.join(annotation_dir, a5ss_k562_basic)
                    background_center = os.path.join(annotation_dir, a5ss_k562_center)
                    background_extension = os.path.join(annotation_dir, a5ss_k562_extension)
                else:
                    print(event)
            elif cell == 'HepG2':
                if event == 'a3ss':
                    background_all = os.path.join(annotation_dir, a3ss_hepg2_all)
                    background_basic = os.path.join(annotation_dir, a3ss_hepg2_basic)
                    background_center = os.path.join(annotation_dir, a3ss_hepg2_center)
                    background_extension = os.path.join(annotation_dir, a3ss_hepg2_extension)
                elif event == 'a5ss':
                    background_all = os.path.join(annotation_dir, a5ss_hepg2_all)
                    background_basic = os.path.join(annotation_dir, a5ss_hepg2_basic)
                    background_center = os.path.join(annotation_dir, a5ss_hepg2_center)
                    background_extension = os.path.join(annotation_dir, a5ss_hepg2_extension)
                else:
                    print(event)
            else:
                print(cell)

            splicing_prefix = m.get_rnaseq_splicing_prefix_from_rbpname(rnaseq_manifests, rbp, cell)
            if(splicing_prefix == "NO_RNASEQ"): # we don't have an rna seq expt for this clip:
                no_rnaseq.append(uid)
            else:
                positive, negative = m.get_annotations_from_splicing_prefix(
                    annotation_dir, 
                    splicing_prefix,
                    pos_splicing_suffix=pos_splicing_suffix,
                    neg_splicing_suffix=neg_splicing_suffix
                )
                if(positive == None or negative == None):
                    no_rnaseq_yet.append(uid)
                else:
                    if not (rbp in positive and rbp in negative):
                        print(
                            'warning, these dont match: {}, {}, {}'.format(
                                rbp, 
                                os.path.basename(positive),
                                os.path.basename(negative)
                            )
                        )
                    pos_prefix = os.path.basename(positive).split('-')[0]
                    neg_prefix = os.path.basename(negative).split('-')[0]
                    if not (pos_prefix in rbp and neg_prefix in rbp):
                        print(
                            'warning, these dont match: {}, {}, {}'.format(
                                rbp, 
                                os.path.basename(positive),
                                os.path.basename(negative)
                            )
                        )
                    for r in [r1, r2]:
                        name = os.path.basename(r).replace('.bam','.{}'.format(img_extension))
                        output_filename = os.path.join(
                            output_dir,
                            name
                        )
                        cmd = "python " + density_runner
                        cmd = cmd + " --event {}".format(event)
                        cmd = cmd + " --ipbam {}".format(r)
                        cmd = cmd + " --inputbam {}".format(i)
                        cmd = cmd + " --output {}".format(output_filename)
                        if positive is not None and negative is not None:
                            cmd = cmd + " --annotations {} {} {} {} {} {}".format(
                                positive, negative, background_all, background_basic, background_center, background_extension
                            )
                            cmd = cmd + " --annotation_type {} {} {} {} {} {}".format(
                                'rmats', 'rmats', 'eric', 'eric', 'eric', 'eric'
                            )
                        # cmd = cmd + " --chrom_sizes {}".format(chrom_sizes)
                        cmd = cmd + " --bgnum {}".format(2)
                        cmd = cmd + " --testnum {} {}".format(0, 1)
                        cmds.append(cmd)
            # if(uid == '228'):
            #     print(r1, r2, i, rbp, cell, annotation_dir, splicing_prefix, pos_splicing_suffix, neg_splicing_suffix)
        bash_script_sh = '/projects/ps-yeolab3/bay001/maps/bash_scripts/{}/{}_NR_{}.sh'.format(
            current_date, event, img_extension
        )
        Submitter(
            cmds, 
            "{}_NR_{}".format(event, img_extension), 
            sh=bash_script_sh,
            submit=True,
            array=True,
            walltime='2:00:00',
            queue='home-yeo'
        )
        with open(bash_script_sh.replace('.sh','.missing.txt'), 'w') as o:
            for no in no_rnaseq:
                o.write(
                    '{}\t{}\n'.format(
                        m.get_clip_file_from_uid(clip_df, no)[3],
                        m.get_clip_file_from_uid(clip_df, no)[4],
                    )
                )
            print("\n\nNO SUFFICIENT POSITIVE OR NEGATIVE SIGNIFICANT ANNOTATIONS:")
            for no in no_rnaseq_yet:
                print(m.get_clip_file_from_uid(clip_df, no)[3:]),



