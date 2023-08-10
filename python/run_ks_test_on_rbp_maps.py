import pandas as pd
import numpy as np
import os
import glob
from tqdm import tnrange, tqdm_notebook

runner = '/home/bay001/projects/codebase/rbp-maps/maps/analysis/ks_plots.py'

def get_prefix(f):
    return os.path.basename(f).split('.')[0]

def run_a3ss_ks_tests():
    input_directory = '/projects/ps-yeolab3/bay001/maps/current/a3ss_nr/'
    all_positives = glob.glob(os.path.join(input_directory,"*-longer-isoform-included-upon-knockdown.normalize_and_per_region_subtract.csv"))
    progress = tnrange(len(all_positives))

    for p in all_positives:
        try:
            if "HepG2" in p:
                cell = 'HepG2'
            elif "K562" in p:
                cell = 'K562'
            else:
                print("Warning, cell not found. defaulting to HepG2")
                cell = 'HepG2'
            n = glob.glob(
                os.path.join(
                    os.path.dirname(p),
                    "{}*{}*-shorter-isoform-included-upon-knockdown.normalize_and_per_region_subtract.csv".format(get_prefix(p), cell)
                )
            )[0]
            c = glob.glob(
                os.path.join(
                    os.path.dirname(p),
                    "{}*{}*-mixed-psi-isoform-in-majority-of-controls.normalize_and_per_region_subtract.csv".format(get_prefix(p), cell)
                )
            )[0]
            output_d_file = os.path.join(os.path.dirname(p), 'ks/{}.ks_nl10dvalues.txt'.format(get_prefix(p)))
            output_p_file = os.path.join(os.path.dirname(p), 'ks/{}.ks_nl10pvalues.txt'.format(get_prefix(p)))
            for condition in [p, n]:
                cmd = 'python {} '.format(runner)
                cmd = cmd + '--input {} '.format(condition)
                cmd = cmd + '--control {} '.format(c)
                cmd = cmd + '--p-output {} '.format(output_p_file)
                cmd = cmd + '--d-output {} '.format(output_d_file)
                if not os.path.exists(output_p_file):
                    get_ipython().system(' $cmd')
        except Exception as e:
            print(e, p)
        progress.update(1)

run_a3ss_ks_tests()

def run_a5ss_ks_tests():
    input_directory = '/projects/ps-yeolab3/bay001/maps/current/a5ss_nr/'
    all_positives = glob.glob(os.path.join(input_directory,"*-longer-isoform-included-upon-knockdown.normalize_and_per_region_subtract.csv"))
    progress = tnrange(len(all_positives))
    for p in all_positives:
        try:
            if "HepG2" in p:
                cell = 'HepG2'
            elif "K562" in p:
                cell = 'K562'
            else:
                print("Warning, cell not found. defaulting to HepG2")
                cell = 'HepG2'
            n = glob.glob(
                os.path.join(
                    os.path.dirname(p),
                    "{}*{}*-shorter-isoform-included-upon-knockdown.normalize_and_per_region_subtract.csv".format(get_prefix(p), cell)
                )
            )[0]
            c = glob.glob(
                os.path.join(
                    os.path.dirname(p),
                    "{}*{}*-mixed-psi-isoform-in-majority-of-controls.normalize_and_per_region_subtract.csv".format(get_prefix(p), cell)
                )
            )[0]
            output_d_file = os.path.join(os.path.dirname(p), 'ks/{}.ks_nl10dvalues.txt'.format(get_prefix(p)))
            output_p_file = os.path.join(os.path.dirname(p), 'ks/{}.ks_nl10pvalues.txt'.format(get_prefix(p)))
            for condition in [p, n]:
                cmd = 'python {} '.format(runner)
                cmd = cmd + '--input {} '.format(condition)
                cmd = cmd + '--control {} '.format(c)
                cmd = cmd + '--p-output {} '.format(output_p_file)
                cmd = cmd + '--d-output {} '.format(output_d_file)
                if not os.path.exists(output_p_file):
                    get_ipython().system(' $cmd')
        except Exception as e:
            print(e, p)
        progress.update(1)

run_a5ss_ks_tests()

def run_ri_ks_tests():
    input_directory = '/projects/ps-yeolab3/bay001/maps/current/ri_nr/'
    all_positives = glob.glob(os.path.join(input_directory,"*-included-upon-knockdown.normalize_and_per_region_subtract.csv"))
    progress = tnrange(len(all_positives))
    for p in all_positives:
        try:
            if "HepG2" in p:
                cell = 'HepG2'
            elif "K562" in p:
                cell = 'K562'
            else:
                print("Warning, cell not found. defaulting to HepG2")
                cell = 'HepG2'
            n = glob.glob(
                os.path.join(
                    os.path.dirname(p),
                    "{}*{}*-excluded-upon-knockdown.normalize_and_per_region_subtract.csv".format(get_prefix(p), cell)
                )
            )[0]
            c = glob.glob(
                os.path.join(
                    os.path.dirname(p),
                    "{}*-greater-than-50-percent-retained-and-spliced-combined.normalize_and_per_region_subtract.csv".format(get_prefix(p))
                )
            )[0]
            output_d_file = os.path.join(os.path.dirname(p), 'ks/{}.ks_nl10dvalues.txt'.format(get_prefix(p)))
            output_p_file = os.path.join(os.path.dirname(p), 'ks/{}.ks_nl10pvalues.txt'.format(get_prefix(p)))
            for condition in [p, n]:
                cmd = 'python {} '.format(runner)
                cmd = cmd + '--input {} '.format(condition)
                cmd = cmd + '--control {} '.format(c)
                cmd = cmd + '--p-output {} '.format(output_p_file)
                cmd = cmd + '--d-output {} '.format(output_d_file)
                if not os.path.exists(output_p_file):
                    get_ipython().system(' $cmd')
        except Exception as e:
            print(e, p)
        progress.update(1)

run_ri_ks_tests()

def run_se_ks_tests():
    input_directory = '/projects/ps-yeolab3/bay001/maps/current/se_nr/'
    all_positives = glob.glob(os.path.join(input_directory,"*.positive.nr.normalize_and_per_region_subtract.csv"))
    progress = tnrange(len(all_positives))
    for p in all_positives:
        try:
            if "HepG2" in p:
                cell = 'HepG2'
            elif "K562" in p:
                cell = 'K562'
            else:
                print("Warning, cell not found. defaulting to HepG2")
                cell = 'HepG2'
            n = glob.glob(
                os.path.join(
                    os.path.dirname(p),
                    "{}*{}*.negative.nr.normalize_and_per_region_subtract.csv".format(get_prefix(p), cell)
                )
            )[0]
            c = glob.glob(
                os.path.join(
                    os.path.dirname(p),
                    # "{}*{}*.nSE_0.5.normalize_and_per_region_subtract.csv".format(get_prefix(p), cell) # this was renamed somehow
                    "{}*{}*.nSEall_0.normalize_and_per_region_subtract.csv".format(get_prefix(p), cell.lower())
                )
            )[0]
            
            output_d_file = os.path.join(os.path.dirname(p), 'ks/{}.ks_nl10dvalues.txt'.format(get_prefix(p)))
            output_p_file = os.path.join(os.path.dirname(p), 'ks/{}.ks_nl10pvalues.txt'.format(get_prefix(p)))
            for condition in [p, n]:
                cmd = 'python {} '.format(runner)
                cmd = cmd + '--input {} '.format(condition)
                cmd = cmd + '--control {} '.format(c)
                cmd = cmd + '--p-output {} '.format(output_p_file)
                cmd = cmd + '--d-output {} '.format(output_d_file)
                if not os.path.exists(output_p_file):
                    get_ipython().system(' $cmd')
        except Exception as e:
            print(e, p, n)
        progress.update(1)

run_se_ks_tests()

