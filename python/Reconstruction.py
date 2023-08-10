import matplotlib.pyplot as plt
import wfdb

from cardiovector import reconstruction as rec, preprocessing as prep, plotting

wfdb.dl_database('ptbdb', dl_dir='data/',
                records=['patient001/s0010_re'],
                overwrite=False)

raw_record = wfdb.rdrecord('data/patient001/s0010_re', physical=False)
record = prep.remove_baseline_wandering(raw_record)
record = prep.recslice(record, sampto=3000)
wfdb.plot_wfdb(record, figsize=(8, 12))

plotting.plotvcg(record, signals=['vx', 'vy', 'vz'],
                 plot=['3d', 'frontal']);

help(rec.kors_vcg)

kors_record = rec.kors_vcg(record)
plotting.plotrecs([record, kors_record], 
                  signals=['vx', 'vy', 'vz'], labels=['frank', 'kors'], 
                  fig_kw={'figsize': (12,7)});

help(rec.idt_vcg)

idt_record = rec.idt_vcg(record)
plotting.plotrecs([record, idt_record], 
                  signals=['vx', 'vy', 'vz'], labels=['frank', 'idt'], 
                  fig_kw={'figsize': (12,7)});

help(rec.pca_vcg)

pca_record = rec.pca_vcg(record)
plotting.plotrecs([record, pca_record], 
                  signals=['vx', 'vy', 'vz'], labels=['frank', 'pca'], 
                  fig_kw={'figsize': (12,7)});

