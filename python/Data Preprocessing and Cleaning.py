import os
from cleaner import clean_years

DATA_DIR = 'data'
DIR_XML_DATA = os.path.join(DATA_DIR, 'GDL')
DIR_OUTPUT_PKL_FILES = os.path.join(DATA_DIR, 'GDL_pkl')
year_start = 1805
year_end = 1825
clean_years(DIR_XML_DATA, DIR_OUTPUT_PKL_FILES, year_start, year_end, False)

from dictFunctions import create_dictionary, load_dictionary, load_dictionaries
from dictFunctions import clean_dict_by_occ, merge_dictionaries
from correctText import clean_dictionary, cleanAndSaveArticles

ranges = [(range(year,year+1)) for year in range(year_start,year_end+1)]

DIR_OUTPUT_DICTIONARIES = os.path.join(DATA_DIR, 'GDL_dict')
for range_values in ranges:
    create_dictionary(range_values, DIR_OUTPUT_PKL_FILES, DIR_OUTPUT_DICTIONARIES)

interval = 10
years = range(year_start, year_end, interval)
occs_clean = [2,3,3,3,5,5,5,5]

for year in years:
    dict_10years = load_dictionaries(DIR_OUTPUT_DICTIONARIES, year, year + interval)
    dict_10years_cleaned = [clean_dict_by_occ(dictio, occs_clean) for dictio in dict_10years]
    fileName = os.path.join(DIR_OUTPUT_DICTIONARIES, str(year) + '-' + str(year + interval) + '.pkl')
    merge_dictionaries(dict_10years_cleaned, fileName)
    print(str(year) + '-' + str(year + interval) + ' done')

dictPath = os.path.join(DIR_OUTPUT_DICTIONARIES, str(year_start) + '-' + str(year_start + interval) + '.pkl')
dictionnaries = clean_dictionary(dictPath)

DIR_OUTPUT_PKL_FILES_CLEANED = os.path.join(DATA_DIR, 'GDL_cleaned')
cleanAndSaveArticles(DIR_OUTPUT_PKL_FILES, DIR_OUTPUT_PKL_FILES_CLEANED, year_start, year_start+1, dictionnaries)



