import urllib2
import json
import numpy as np
from unittests.validate_annotations import test_annotations
import warnings
from morty.converter import Converter
from morty.pitchdistribution import PitchDistribution
from matplotlib import pyplot as plt

# load master data
anno_master = json.load(open('../annotations.json'))

# load v3.0.0-alpha data
anno_alpha_url = 'https://raw.githubusercontent.com/MTG/otmm_tonic_dataset/v3.0.0-alpha/annotations.json'
response = urllib2.urlopen(anno_alpha_url)
anno_alpha = json.load(response)

test_annotations(anno_master)

try:
    test_annotations(anno_alpha)
except AssertionError as err:
    # get the number of mismatches
    print err
    num_err = [int(s) for s in err.args[0].split() if s.isdigit()][0]
    inconsistent_percent = float(num_err) * 100 / (2007-1589)  # taken from the penultimate warning
    print("The human annotators show inconsistencies in {:.1f}% of "
          "the {:d} tested recordings.".format(inconsistent_percent, 2007-1589))
    

rec_stats = {}
cent_dev = []
for aa_key, aa_val in anno_alpha.items():
    try:
        # get the relevant recording entry in master
        am_val = anno_master[aa_key]
        rec_stats[aa_key] = {'num_deleted_anno': 0, 'status': 'kept', 
                             'num_added_anno': 0, 'num_unchanged_anno': 0,
                             'num_modified_anno': 0, 'num_auto_anno': 0,
                             'verified': am_val['verified']}
        
        # note automatic annotations in master; they did not exist in v3.0.0-alpha
        for jj, am_anno in reversed(list(enumerate(am_val['annotations']))):
            if 'jointanalyzer' in am_anno['source']:
                rec_stats[aa_key]['num_auto_anno'] += 1
                am_val['annotations'].pop(jj)
        
        # start comparison from v3.0.0 to master
        for ii, aa_anno in reversed(list(enumerate(aa_val['annotations']))):
            passed_break = False
            for jj, am_anno in reversed(list(enumerate(am_val['annotations']))):
                if aa_anno['source'] == am_anno['source']:  # annotation exists
                    # unchanged anno; allow a change less than 0.051 Hz due to 
                    # decimal point rounding
                    if abs(aa_anno['value'] - am_anno['value']) < 0.06:
                        rec_stats[aa_key]['num_unchanged_anno'] += 1
                    else:  # modified anno (by a human verifier)
                        rec_stats[aa_key]['num_modified_anno'] += 1
                        
                        # find the introduced octave-wrapped deviation
                        temp_dev = Converter.hz_to_cent(
                            aa_anno['value'], am_anno['value'])  # hz to cent conversion
                        temp_dev = temp_dev % 1200  # octave wrap
                        temp_dev = min(temp_dev, 1200-temp_dev)  # get minimum distance
                        cent_dev.append(temp_dev)
                        
                    # pop annotations
                    am_val['annotations'].pop(jj)
                    aa_val['annotations'].pop(ii)
                    break
                    
        # the remainders are human addition and deletions
        rec_stats[aa_key]['num_added_anno'] = len(am_val['annotations'])
        rec_stats[aa_key]['num_deleted_anno'] = len(aa_val['annotations'])
                              
    except KeyError as kerr:  # removed 
        rec_stats[kerr.args[0]] = {'num_deleted_anno':len(aa_val['annotations']),
                                   'status': 'removed', 'num_added_anno': 0, 
                                   'num_modified_anno': 0, 'num_unchanged_anno': 0,
                                   'num_auto_anno': 0, 'verified': True}

new_recs = set(anno_master.keys()) - set(anno_alpha.keys())
for am_key in new_recs:
    am_val = anno_master[am_key]
    rec_stats[am_key] = {'num_deleted_anno': 0, 'status': 'new', 
                         'num_added_anno': 0, 'num_unchanged_anno': 0,
                         'num_modified_anno': 0, 'num_auto_anno': 0,
                         'verified': am_val['verified']}

    # note automatic annotations; they did not exist in v3.0.0-alpha
    for jj, am_anno in reversed(list(enumerate(am_val['annotations']))):
        if 'jointanalyzer' in am_anno['source']:
            rec_stats[am_key]['num_auto_anno'] += 1
            am_val['annotations'].pop(jj)
    
    # the remainders are human additions
    rec_stats[am_key]['num_added_anno'] = len(am_val['annotations'])

# removed 
rm_recs_in_json = json.load(open('../removed.json')).keys()

# TODO add statistics
num_removed_rec = 0
num_new_rec = 0

num_changed_rec = 0  # num recordings with changes, incl. automatic annotations
num_human_changed_rec = 0  # num recordings with human changes

num_anno = 0  # total number of annotations
num_verified_anno = 0  # total number of verified annotations
num_human_verified_anno = 0  # totola number of annotations verified by humans

num_additions = 0  # number of added annotations
num_deletions = 0  # number of deleted annotations
num_modifications = 0  # number of modified annotations
num_unchanged = 0  # number of unchanged annotations
num_auto = 0  # number of automatic annotations

num_rec_add = 0  # number of recordings with additions
num_rec_del = 0  # number of recordings with deletions
num_rec_mod = 0  # number of recordings with modification
num_rec_auto = 0  # number of recordings with automatic annotations

for rk, rs in rec_stats.items():
    # get the number of removed and new recordings
    if rs['status'] == 'removed':
        num_removed_rec += 1
        if rk not in rm_recs_in_json:  # verify they are listed in removed.json
            warnings.warn('%s is removed but not listed in removed.json' % rk)
    elif rs['status'] == 'new':
        num_new_rec += 1
        
    num_anno += (rs['num_added_anno'] + rs['num_auto_anno'] + 
                 rs['num_modified_anno'] + rs['num_unchanged_anno'])
    
    # how many recordings have changed
    if any([rs['num_added_anno'], rs['num_auto_anno'], 
            rs['num_deleted_anno'], rs['num_modified_anno']]):
        num_changed_rec += 1
        num_verified_anno += (rs['num_added_anno'] + rs['num_auto_anno'] + 
                              rs['num_modified_anno'] + rs['num_unchanged_anno'])
        
    # how many recordings have changed only by humans
    if any([rs['num_added_anno'], rs['num_deleted_anno'], rs['num_modified_anno']]):
        num_human_changed_rec += 1
        num_human_verified_anno += (rs['num_added_anno'] + rs['num_auto_anno'] + 
                                   rs['num_modified_anno'] + rs['num_unchanged_anno'])
        if not rs['verified']:
            warnings.warn("%s has changes but verified flag is False" % rk)
        
    # how many automatic annotations in how many recordings
    num_auto += rs['num_auto_anno']
    num_rec_auto += rs['num_auto_anno'] > 0
    
    # how many annotation modifications/additions/deletions in how many recordings
    num_additions += rs['num_added_anno']
    num_rec_add += rs['num_added_anno'] > 0
    num_deletions += rs['num_deleted_anno']
    num_rec_del += rs['num_deleted_anno'] > 0
    num_modifications += rs['num_modified_anno']
    num_rec_mod += rs['num_modified_anno'] > 0
    
    # how many unchanged annotations
    num_unchanged += rs['num_unchanged_anno']
    
    # distribution of human frequency modifications
    
# print 
print('In master, there are %d annotations in total in %d recordings.' 
      % (num_anno, len(anno_master)))
print('Since v3.0.0-alpha, %d recordings are removed and %d new recordings are added.'
      % (num_removed_rec, num_new_rec))

print('%d recordings are changed (incl. automatic annotations). '
      '%d of the annotations are verified in these recordings.' 
      % (num_changed_rec, num_verified_anno))

print('%d annotations in %d recordings are changed by humans in total. '
      '%d of the annotations are verified by humans in these recordings.' 
      % (num_additions + num_deletions + num_modifications, 
         num_human_changed_rec, num_human_verified_anno))

print('%d annotations are added to %d recordings by humans.' %(num_additions, num_rec_add))
print('%d annotations are deleted from %d recordings by humans.' %(num_deletions, num_rec_del))
print('%d annotations are modified in %d recordings by humans.' %(num_modifications, num_rec_mod))

print('%d automatic annotations are added to %d recordings.' %(num_auto, num_rec_auto))

dev_dist = PitchDistribution.from_cent_pitch(
    cent_dev, step_size=10, kernel_width=0, norm_type=None)
dev_dist.bar()
plt.title('Distribution of absolute deviation from \naverage of annotation pitch-class')
plt.xlabel('Absolute deviation (cents)')
plt.xlim([min(dev_dist.bins), max(dev_dist.bins) + dev_dist.step_size/2])
plt.xticks(np.arange(0, 600, 50))
plt.show()

print("%d of %d modifications were done within a 20 cent octave-wrapped window."
      % (sum(c < 20 for c in cent_dev), len(cent_dev)))
print("%d of %d modifications were done within a 50 cent octave-wrapped window."
       % (sum(c < 50 for c in cent_dev), len(cent_dev)))

