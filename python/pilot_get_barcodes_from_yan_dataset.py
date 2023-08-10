import pandas as pd

with open(
    '/projects/ps-yeolab3/jobs/cellranger/takeda/2017-10-06_YanSong_TPH1YS567/results/YS5_expression.csv',
    'r'
) as f:
    barcodes = (f.readline().split(','))[1:] # first line is empty
barcodes[:5]

with open(
    '/home/bay001/projects/codebase/convertTSV/data/YanSong_TPH1YS567_20171006_barcodes.tsv', 'w'
) as o:
    for barcode in barcodes:
        o.write("{}\n".format(barcode.split('-1')[0]))



