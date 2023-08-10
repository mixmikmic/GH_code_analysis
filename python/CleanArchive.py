#import sys, os
#sys.path.append("/data/shared/Delphes/Software/")
archive_dir = "/data/shared/Delphes/keras_archive_2_19/"
from CMS_Deep_Learning.storage.archiving import *
datas = get_all_data(archive_dir)
print(len(datas))
for data in datas:
    data.remove_from_archive()

from CMS_Deep_Learning.storage.archiving import *
archive_dir = "/data/shared/Delphes/keras_archive_danny/"
def get_trial_dps(trial, data_type="train"):
    from CMS_Deep_Learning.storage.archiving import DataProcedure
    if (data_type == "val"):
        proc = [DataProcedure.from_json(trial.archive_dir, t) for t in trial.val_procedure]
        # num_samples = trial.nb_val_samples
    elif (data_type == "train"):
        proc = [DataProcedure.from_json(trial.archive_dir, t) for t in trial.train_procedure]
        # num_samples = trial.samples_per_epoch
    return proc
trial = KerasTrial.find_by_hashcode(archive_dir, "d93e69176caef48cd3f187e137fb03c305b5f156")
trial.summary()
print(get_trial_dps(trial))
train = get_trial_dps(trial, data_type="train")
val = get_trial_dps(trial, data_type="train")
datas = train + val
for data in datas:
    data.remove_from_archive()
trial.remove_from_archive()





