from collections import defaultdict
import re
import os
import glob
import csv

# code moved to external file so that FME can also read it
from DCF_Parser_Main import parseDCF

# inDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\DHS_Automation\Acquisition\20160623_Updates\in'
inDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\DHS_Automation\Acquisition\All\437'
inDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\DHS_Automation\Acquisition\All_MR\some_manual_mr_downloads_cos_dhs_website_is_shit'
inDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\DHS_Automation\Acquisition\20170328_Updates'
#outDir =  r'\\map-fs1.ndph.ox.ac.uk\map_data\DHS_Automation\Acquisition\20160623_Updates\parsed'
outDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\DHS_Automation\Acquisition\All\437'
outDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\DHS_Automation\Acquisition\All_MR\parsed'
outDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\DHS_Automation\Acquisition\20170328_Updates\parsed'

# Specify what columns the CSV should have - these must be things that the DCF parser generates
reqfieldnames = ['ItemType', 'FileCode','RecordName','RecordTypeValue','RecordLabel','Name','Label',
                 'Start','Len','Occurrences','ZeroFill', 'DecimalChar', 'Decimal', 'FMETYPE']
valfieldnames = ['FileCode','Name','Value','ValueDesc', 'ValueType']
relfieldnames = ['FileCode', 'RelName', 'PrimaryTable', 'PrimaryLink', 'SecondaryTable', 'SecondaryLink']

inputDCFs = glob.glob(os.path.join(inDir,'*','*.dcf'))

if not os.path.exists(outDir):
    os.makedirs(outDir)
else:
    assert(os.path.isdir(outDir))

for inputDCF in inputDCFs:
    print inputDCF
    # parse it!
    parsedDCFItems, parsedDCFRelations = parseDCF(inputDCF, expandRanges="All")
    
    inBase = os.path.extsep.join(os.path.basename(inputDCF).split(os.path.extsep)[:-1])
    outBase = inBase + '.FlatRecordSpec.csv'
    outValsBase = inBase + '.FlatValuesSpec.csv'
    outRelsBase = inBase + '.RelationshipsSpec.csv'
    outFileName = os.path.join(outDir,outBase)
    outValsFileName = os.path.join(outDir, outValsBase)
    outRelsFileName = os.path.join(outDir, outRelsBase)
    
    with open(outFileName, 'wb') as fout, open(outValsFileName, 'wb') as fValsOut,             open(outRelsFileName, 'wb') as fRelsOut:
        wri = csv.writer(fout)
        wri.writerow(reqfieldnames)
        wriVals = csv.writer(fValsOut)
        wriVals.writerow(valfieldnames)
        wriRels = csv.writer(fRelsOut)
        wriRels.writerow(relfieldnames)
        for item in parsedDCFItems:
            item['FMETYPE'] = "fme_char({0!s})".format(item['Len'])
            # write the row using the fieldnames given in reqfieldnames
            # not all items have "occurrences", "range_low_value", etc so write blank value if not
            wri.writerow([item[k] if item.has_key(k) else '' for k in reqfieldnames])
            # write the value sets to a separate file
            if 'Values' in item and len(item['Values'])>0:
                vals = item['Values']
                for val in vals:
                    wriVals.writerow([item['FileCode'],item['Name'], val[0], val[1], val[2]])
        for item in parsedDCFRelations:
            wriRels.writerow([item[k] if item.has_key(k) else '' for k in relfieldnames])
            

# Not part of the parsing - a bit of code to apply before we do anything else to 
# rename downloaded files to include survey id number. Assumes they have been downloaded into 
# subdirectories named with the id number.
allFiles = glob.glob(os.path.join(inDir,'*','*'))
for fn in allFiles:
    if str.lower(fn).find('.zip') != -1:
        continue
    basename = os.path.basename(fn)
    dirname = os.path.dirname(fn)
    idname = os.path.basename(dirname)
    newname = idname+'.'+basename
    newpath = os.path.join(dirname,newname)
    #print fn
    #print newpath
    os.rename(fn,newpath)

levelsInserts = []
recordsInserts = []
itemsInserts = []
valuesInserts = []

# i think a straightforward bit of sql formatting will do here, nobody malicious will get chance to run...

# levels is straightforward, just name and label
for name, label in allLevels.iteritems():
    insertLevelsSQL = 'INSERT INTO dhs_levels '    '(record_name, record_label)'     ' VALUES '    '("{0!s}", "{1!s}");'.format(name, label)
    levelsInserts.append(insertLevelsSQL)
    
# records may be country specific, impute this from the presence of a word followed by "specific"
# in the label. E.g. "Country specific", "Survey specific"
for name, label in allRecords.iteritems():
    m = re.match('^\w+ specific', label)
    if m:
        specificText = m.group(0)
    else:
        specificText = "No"
        
    insertRecordsSQL = 'INSERT INTO dhs_records '    '(record_name, record_label, c_or_s_specific)'     ' VALUES '    '("{0!s}", "{1!s}", "{2!s}");'.format(name, label, specificText)
    recordsInserts.append(insertRecordsSQL)
    
# items is the main thingy
for item in allItems:
    insertItemsSQL = 'INSERT INTO dhs_recodes '    '(level_id, record_id, recode_id, recode_description, start, len, data_type, item_type, range_low_value, range_high_value)'    ' VALUES '    '("{0!s}", "{1!s}", "{2!s}", "{3!s}", {4!s}, {5!s}, {6!s}, {7!s}, {8!s}, {9!s});'.format(
        item['LevelName'], item['RecordName'], item['Name'], item['Label'], item['Start'], item['Len'], 
        '"A"', '"B"', # TODO change these
        item['Range_Low_Value'] if item.has_key('Range_Low_Value') else '',
        item['Range_High_Value'] if item.has_key('Range_High_Value') else '',
    )
    itemsInserts.append(insertItemsSQL)
    if item.has_key('Values') and len(item['Values'])>0:
        for valtuple in item['Values']:
            insertValueSQL = 'INSERT INTO dhs_recode_values '            '(recode_id, value_code, value_description)'             ' VALUES '            '("{0!s}", "{1!s}", "{2!s}");'.format(
                item['Name'], valtuple[0], valtuple[1])
            valuesInserts.append(insertValueSQL)
#print insertSQL



