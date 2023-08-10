import urllib
import urllib2
import xlrd
import csv
import pandas as pd
import numpy as np
import re
from itertools import islice

urls = {
    'education.xls': 'https://www.ers.usda.gov/webdocs/DataFiles/CountyLevel_Data_Sets_Download_Data__18026//Education.xls',
    'population_estimates.xls': 'https://www.ers.usda.gov/webdocs/DataFiles/CountyLevel_Data_Sets_Download_Data__18026//PopulationEstimates.xls',
    'unemployment.xls': 'https://www.ers.usda.gov/webdocs/DataFiles/CountyLevel_Data_Sets_Download_Data__18026//Unemployment.xls',
    'income.xlsx': 'https://www.ers.usda.gov/webdocs/DataFiles/Rural_Atlas_Download_the_Data__18022//RuralAtlasData13.xlsx'
    }

# downloads xls files from census.gov
def download_census_files(): 
    testfile = urllib.URLopener()
    for key in urls:
        testfile.retrieve(urls[key], key)

# converts xls files to csv
def csv_from_xls(): 
    for key in urls:
        wb = xlrd.open_workbook(key)
        if key == 'education.xls':
            sh = wb.sheet_by_name('Education 1970 to 2015')
            csv_file = open('education.csv', 'wb')
        elif key == 'population_estimates.xls':
            sh = wb.sheet_by_name('Population Estimates 2010-2015')
            csv_file = open('population_estimates.csv', 'wb')
        elif key == 'unemployment.xls':
            sh = wb.sheet_by_name('Unemployment Med HH Inc ')
            csv_file = open('unemployment.csv', 'wb')
        elif key == 'income.xlsx':
            sh = wb.sheet_by_name('Income')
            csv_file = open('income.csv', 'wb')
        
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        
        for rownum in range(sh.nrows):
            csv.writer(csv_file, quoting=csv.QUOTE_ALL).writerow(
                 list(x.encode('utf-8') if type(x) == type(u'') else x
                      for x in sh.row_values(rownum)))

        csv_file.close()

# This script cleans the "Income" tab of the "RuralAtlasData13.csv" file available at census.gov. It pulls out only the 2014 income data
def income_clean(): 
    f_in = open('income.csv', 'r')
    f_out = open('Income_pivot.csv', 'w')
    
    #--add spaces
    reader_in = csv.reader(f_in, delimiter=',')
    writer_out = csv.writer(f_out, delimiter=',')

    line_num = 0
    year_dict = {}

    for line in islice(reader_in, 0, None):
        #--put at end
        line_num +=1
        if line_num == 1:
            line_out = ["fips", line[3], line[4], line[5], line[6], line[7], line[8]]
            line_dict = list(enumerate(line))
            for key, value in line_dict:
                year = re.search('20\d+(?!.*20\d+)'.decode('utf-8'), value.decode('utf-8'), re.I | re.U)
                if year:
                    year_dict[key] = year.group(0)
            writer_out.writerow(line_out)
        else:
            line_out = [line[0], line[3], line[4], line[5], line[6], line[7], line[8]]
            writer_out.writerow(line_out)

    f_in.close()
    f_out.close()

# This script cleans the "Education.csv" file available at census.gov. It pulls out only the 2010 - 2014 education stats

def education_clean(): 
    f_in = open('education.csv', 'r')
    f_out = open('education_pivot.csv', 'w')

    reader_in = csv.reader(f_in, delimiter=',')
    writer_out = csv.writer(f_out,delimiter=',')

    line_num = 0
    year_dict = {}

    for line in islice(reader_in, 4, None):
        line_num +=1
        if line_num == 1:
            line_out = ["fips", line[43], line[44], line[45], line[46]]
            line_dict = list(enumerate(line))
            for key, value in line_dict:
                year = re.search('20\d+(?!.*20\d+)'.decode('utf-8'), value.decode('utf-8'), re.I | re.U)
                if year:
                    year_dict[key] = year.group(0)
            writer_out.writerow(line_out)
        else:
            line_out = [line[0], line[43], line[44], line[45], line[46]]
            writer_out.writerow(line_out)

    f_in.close()
    f_out.close()

# This script cleans the "PopulationEstimates.csv" file available at census.gov. It pulls out only the 2015 population data
def population_clean(): 
    f_in = open('population_estimates.csv', 'r')
    f_out = open('population_estimates_pivot.csv', 'w')

    reader_in = csv.reader(f_in, delimiter=',')
    writer_out = csv.writer(f_out,delimiter=',')

    line_num = 0
    year_dict = {}

    for line in islice(reader_in, 2, None):
        line_num +=1
        if line_num == 1:
            line_out = ["fips", line[15]]
            line_dict = list(enumerate(line))
            for key, value in line_dict:
                year = re.search('20\d+(?!.*20\d+)'.decode('utf-8'), value.decode('utf-8'), re.I | re.U)
                if year:
                    year_dict[key] = year.group(0)
            writer_out.writerow(line_out)
        else:
            line_out = [line[0], line[15]]
            writer_out.writerow(line_out)

    f_in.close()
    f_out.close()

# This script cleans the "Unemployment.csv" file available at census.gov. It pulls out only the 2015 population data
def unemployment_clean(): 
    f_in = open('unemployment.csv', 'r')
    f_out = open('unemployment_pivot.csv', 'w')

    reader_in = csv.reader(f_in, delimiter=',')
    writer_out = csv.writer(f_out,delimiter=',')

    line_num = 0
    year_dict = {}

    for line in islice(reader_in, 7, None):
        line_num +=1
        if line_num == 1:
            line_out = ["fips", line[42]]
            line_dict = list(enumerate(line))
            for key, value in line_dict:
                year = re.search('20\d+(?!.*20\d+)'.decode('utf-8'), value.decode('utf-8'), re.I | re.U)
                if year:
                    year_dict[key] = year.group(0)
            writer_out.writerow(line_out)
        else:
            line_out = [line[0], line[42]]
            writer_out.writerow(line_out)

    f_in.close()
    f_out.close()


def combine_files():
    # Grab google drive files
    election_results = urllib2.urlopen('https://docs.google.com/spreadsheets/d/1jwysAyxJKdP8fIWJddlrlBSZUmnCHY6AF1n52WoDnx8/export?format=csv')
    age = urllib2.urlopen('https://docs.google.com/spreadsheets/d/1-5KonTNEDRPMrLnXyfBea6vjp6_GR7On735dqsgpkcg/export?format=csv')
    gender_and_race = urllib2.urlopen('https://docs.google.com/spreadsheets/d/1SumP-nF1wDauhxno1m7KkYeHnGHArqXEzDH-_k9gRQ8/export?format=csv')
    population_density = urllib2.urlopen('https://docs.google.com/spreadsheets/d/1FvknWOv5TBuCZnZCcFc3Sk4CJIJaiAz0Rz2w3ofOSW8/export?format=csv')
    religion = urllib2.urlopen('https://docs.google.com/spreadsheets/d/1JPl0PKgJHjemsX0To5TMRo3oXIo6PKAndqeYqv7drUI/export?format=csv')
    drug_deaths = urllib2.urlopen('https://docs.google.com/spreadsheets/d/1YcIefCdMs1wYhHOJblC6IyaQ4ko5kVYBUSUd9OG9SKU/export?format=csv')
    
    # Create dataframes from all clean csvs
    election_results_df = pd.read_csv(election_results)
    age_df = pd.read_csv(age)
    gender_and_race_df = pd.read_csv(gender_and_race)
    population_density_df = pd.read_csv(population_density)
    religion_df = pd.read_csv(religion)
    education_df = pd.read_csv('education_pivot.csv')
    unemployment_df = pd.read_csv('unemployment_pivot.csv')
    unemployment_df = unemployment_df.convert_objects(convert_numeric=True)
    population_estimates_df = pd.read_csv('population_estimates_pivot.csv')
    income_df = pd.read_csv('income_pivot.csv')
    drug_deaths_df = pd.read_csv(drug_deaths)
    
    # Replace nulls with 0 in religion_df
    religion_df = religion_df.fillna(0)
    
    # Join data sets on fips county code
    combined = (election_results_df.merge
                    (education_df,on='fips', how='outer').merge
                    (unemployment_df,on='fips', how='outer').merge
                    (population_estimates_df,on='fips', how='outer').merge
                    (religion_df,on='fips', how='outer').merge
                    (income_df,on='fips', how='outer').merge
                    (age_df,on='fips', how='outer').merge
                    (gender_and_race_df,on='fips', how='outer').merge
                    (population_density_df,on='fips', how='outer')
                )
    
    #remove state-level fips
    #combined = combined[combined.geo_name != 'Alaska']
    #combined = combined[combined.geo_name != 'District of Columbia']
    combined = combined[pd.notnull(combined['geo_name'])]
    
    #feature engineering - rates from raw values
    combined["age_total_pop"] = (combined["1"] +
                                 combined["2"] +
                                 combined["3"] +
                                 combined["4"] +
                                 combined["5"] +
                                 combined["6"] +
                                 combined["7"] +
                                 combined["8"] +
                                 combined["9"] +
                                 combined["10"] +
                                 combined["11"] +
                                 combined["12"] +
                                 combined["13"] +
                                 combined["14"] +
                                 combined["15"] +
                                 combined["16"] +
                                 combined["17"] +
                                 combined["18"]  )
    combined["0-4_rate"] = combined["1"] / combined["age_total_pop"]
    combined["5-9_rate"] = combined["2"] / combined["age_total_pop"]
    combined["10-14_rate"] = combined["2"] / combined["age_total_pop"]
    combined["15-19_rate"] = combined["4"] / combined["age_total_pop"]
    combined["20-24_rate"] = combined["5"] / combined["age_total_pop"]
    combined["25-29_rate"] = combined["6"] / combined["age_total_pop"]
    combined["30-34_rate"] = combined["7"] / combined["age_total_pop"]
    combined["35-39_rate"] = combined["8"] / combined["age_total_pop"]
    combined["40-44_rate"] = combined["9"] / combined["age_total_pop"]
    combined["45-49_rate"] = combined["10"] / combined["age_total_pop"]
    combined["50-54_rate"] = combined["11"] / combined["age_total_pop"]
    combined["55-59_rate"] = combined["12"] / combined["age_total_pop"]
    combined["60-64_rate"] = combined["13"] / combined["age_total_pop"]
    combined["65-69_rate"] = combined["14"] / combined["age_total_pop"]
    combined["70-74_rate"] = combined["15"] / combined["age_total_pop"]
    combined["75-79_rate"] = combined["16"] / combined["age_total_pop"]
    combined["80-84_rate"] = combined["17"] / combined["age_total_pop"]
    combined["85+_rate"] = combined["18"] / combined["age_total_pop"]
    
    combined["average_age"] = (combined["1"] * 2 +
                             combined["2"] * 7 +
                             combined["3"] * 12 +
                             combined["4"] * 17 +
                             combined["5"] * 22 +
                             combined["6"] * 27 +
                             combined["7"] * 32 +
                             combined["8"] * 37 +
                             combined["9"] * 42 +
                             combined["10"] * 47 +
                             combined["11"] * 52 +
                             combined["12"] * 57 +
                             combined["13"] * 62 +
                             combined["14"] * 67 +
                             combined["15"] * 72 +
                             combined["16"] * 77 +
                             combined["17"] * 82 +
                             combined["18"] * 88) / combined["age_total_pop"]
    
    combined["TOT_MALE_rate"] = combined["TOT_MALE"] / combined["TOT_POP"]
    combined["TOT_FEMALE_rate"] = combined["TOT_FEMALE"] / combined["TOT_POP"]
    combined["WHITE_MALE_rate"] = combined["WA_MALE"] / combined["TOT_POP"]
    combined["WHITE_FEMALE_rate"] = combined["WA_FEMALE"] / combined["TOT_POP"]
    combined["BLACK_MALE_rate"] = combined["BA_MALE"] / combined["TOT_POP"]
    combined["BLACK_FEMALE_rate"] = combined["BA_FEMALE"] / combined["TOT_POP"]
    combined["NATIVE_AMERICAN_MALE_rate"] = combined["IA_MALE"] / combined["TOT_POP"]
    combined["NATIVE_AMERICAN_FEMALE_rate"] = combined["IA_FEMALE"] / combined["TOT_POP"]
    combined["ASIAN_MALE_rate"] = combined["AA_MALE"] / combined["TOT_POP"]
    combined["ASIAN_FEMALE_rate"] = combined["AA_FEMALE"] / combined["TOT_POP"]
    combined["HAWAIIAN_PACIFIC_MALE_rate"] = combined["NA_MALE"] / combined["TOT_POP"]
    combined["HAWAIIAN_PACIFIC_FEMALE_rate"] = combined["NA_FEMALE"] / combined["TOT_POP"]
    combined["MULTI_MALE_rate"] = combined["TOM_MALE"] / combined["TOT_POP"]
    combined["MULTI_FEMALE_rate"] = combined["TOM_FEMALE"] / combined["TOT_POP"]    

    combined["WHITE_rate"] = (combined["WA_MALE"] + combined["WA_FEMALE"]) / combined["Population"]
    combined["BLACK_rate"] = (combined["BA_MALE"] + combined["BA_FEMALE"]) / combined["Population"]
    combined["NATIVE_AMERICAN_rate"] = (combined["IA_MALE"] + combined["IA_FEMALE"]) / combined["Population"]
    combined["HAWAIIAN_PACIFIC_rate"] = (combined["NA_MALE"] + combined["NA_FEMALE"]) / combined["Population"]
    combined["MULTI_rate"] = (combined["TOM_MALE"] + combined["TOM_FEMALE"]) / combined["Population"]

    
    #fill missing values with weighted mean
    combined["Percent of adults with less than a high school diploma, 2011-2015"] = combined["Percent of adults with less than a high school diploma, 2011-2015"].fillna((((combined["Percent of adults with less than a high school diploma, 2011-2015"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Percent of adults with a high school diploma only, 2011-2015"] = combined["Percent of adults with a high school diploma only, 2011-2015"].fillna((((combined["Percent of adults with a high school diploma only, 2011-2015"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Percent of adults completing some college or associate's degree, 2011-2015"] = combined["Percent of adults completing some college or associate's degree, 2011-2015"].fillna((((combined["Percent of adults completing some college or associate's degree, 2011-2015"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Percent of adults with a bachelor's degree or higher, 2011-2015"] = combined["Percent of adults with a bachelor's degree or higher, 2011-2015"].fillna((((combined["Percent of adults with a bachelor's degree or higher, 2011-2015"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Unemployment_rate_2015"] = combined["Unemployment_rate_2015"].fillna((((combined["Unemployment_rate_2015"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["POP_ESTIMATE_2015"] = combined["POP_ESTIMATE_2015"].fillna((((combined["POP_ESTIMATE_2015"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["MedHHInc2014"] = combined["MedHHInc2014"].fillna((((combined["MedHHInc2014"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["PovertyUnder18Pct2014"] = combined["PovertyUnder18Pct2014"].fillna((((combined["PovertyUnder18Pct2014"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["PovertyAllAgesPct2014"] = combined["PovertyAllAgesPct2014"].fillna((((combined["PovertyAllAgesPct2014"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["PerCapitaInc"] = combined["PerCapitaInc"].fillna((((combined["PerCapitaInc"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Deep_Pov_All"] = combined["Deep_Pov_All"].fillna((((combined["Deep_Pov_All"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Deep_Pov_Children"] = combined["Deep_Pov_Children"].fillna((((combined["Deep_Pov_Children"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Density per square mile of land area - Population"] = combined["Density per square mile of land area - Population"].fillna((((combined["Density per square mile of land area - Population"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Density per square mile of land area - Housing units"] = combined["Density per square mile of land area - Housing units"].fillna((((combined["Density per square mile of land area - Housing units"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["Housing units"] = combined["Housing units"].fillna((((combined["Housing units"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
   
    combined["Population"] = combined["Population"].fillna(combined["Population"].mean())
    combined["age_total_pop"] = combined["age_total_pop"].fillna((((combined["age_total_pop"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["0-4_rate"] = combined["0-4_rate"].fillna((((combined["0-4_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["5-9_rate"] = combined["5-9_rate"].fillna((((combined["5-9_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["10-14_rate"] = combined["10-14_rate"].fillna((((combined["10-14_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["15-19_rate"] = combined["15-19_rate"].fillna((((combined["15-19_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["20-24_rate"] = combined["20-24_rate"].fillna((((combined["20-24_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["25-29_rate"] = combined["25-29_rate"].fillna((((combined["25-29_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["30-34_rate"] = combined["30-34_rate"].fillna((((combined["30-34_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["35-39_rate"] = combined["35-39_rate"].fillna((((combined["35-39_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["40-44_rate"] = combined["40-44_rate"].fillna((((combined["40-44_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["45-49_rate"] = combined["45-49_rate"].fillna((((combined["45-49_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["50-54_rate"] = combined["50-54_rate"].fillna((((combined["50-54_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["55-59_rate"] = combined["55-59_rate"].fillna((((combined["55-59_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["60-64_rate"] = combined["60-64_rate"].fillna((((combined["60-64_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["65-69_rate"] = combined["65-69_rate"].fillna((((combined["65-69_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)
    
    combined["70-74_rate"] = combined["70-74_rate"].fillna((((combined["70-74_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["75-79_rate"] = combined["75-79_rate"].fillna((((combined["75-79_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["80-84_rate"] = combined["80-84_rate"].fillna((((combined["80-84_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["85+_rate"] = combined["85+_rate"].fillna((((combined["85+_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["TOT_MALE_rate"] = combined["TOT_MALE_rate"].fillna((((combined["TOT_MALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["TOT_FEMALE_rate"] = combined["TOT_FEMALE_rate"].fillna((((combined["TOT_FEMALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["WHITE_MALE_rate"] = combined["WHITE_MALE_rate"].fillna((((combined["WHITE_MALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["WHITE_FEMALE_rate"] = combined["WHITE_FEMALE_rate"].fillna((((combined["WHITE_FEMALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["BLACK_MALE_rate"] = combined["BLACK_MALE_rate"].fillna((((combined["BLACK_MALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["BLACK_FEMALE_rate"] = combined["BLACK_FEMALE_rate"].fillna((((combined["BLACK_FEMALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["NATIVE_AMERICAN_MALE_rate"] = combined["NATIVE_AMERICAN_MALE_rate"].fillna((((combined["NATIVE_AMERICAN_MALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["NATIVE_AMERICAN_FEMALE_rate"] = combined["NATIVE_AMERICAN_FEMALE_rate"].fillna((((combined["NATIVE_AMERICAN_FEMALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["ASIAN_MALE_rate"] = combined["ASIAN_MALE_rate"].fillna((((combined["ASIAN_MALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["ASIAN_FEMALE_rate"] = combined["ASIAN_FEMALE_rate"].fillna((((combined["ASIAN_FEMALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)

    combined["HAWAIIAN_PACIFIC_MALE_rate"] = combined["HAWAIIAN_PACIFIC_MALE_rate"].fillna((((combined["HAWAIIAN_PACIFIC_MALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["HAWAIIAN_PACIFIC_FEMALE_rate"] = combined["HAWAIIAN_PACIFIC_FEMALE_rate"].fillna((((combined["HAWAIIAN_PACIFIC_FEMALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["MULTI_MALE_rate"] = combined["MULTI_MALE_rate"].fillna((((combined["MULTI_MALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum())))
    combined["MULTI_FEMALE_rate"] = combined["MULTI_FEMALE_rate"].fillna((((combined["MULTI_FEMALE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)

    combined["Amish"] = combined["Amish"].fillna((((combined["Amish"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Buddhist"] = combined["Buddhist"].fillna((((combined["Buddhist"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Catholic"] = combined["Catholic"].fillna((((combined["Catholic"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Christian Generic"] = combined["Christian Generic"].fillna((((combined["Christian Generic"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Eastern Orthodox"] = combined["Eastern Orthodox"].fillna((((combined["Eastern Orthodox"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Hindu"] = combined["Hindu"].fillna((((combined["Hindu"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Jewish"] = combined["Jewish"].fillna((((combined["Jewish"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Mainline Christian"] = combined["Mainline Christian"].fillna((((combined["Mainline Christian"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Mormon"] = combined["Mormon"].fillna((((combined["Mormon"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Muslim"] = combined["Muslim"].fillna((((combined["Muslim"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Non-Catholic Christian"] = combined["Non-Catholic Christian"].fillna((((combined["Non-Catholic Christian"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Other"] = combined["Other"].fillna((((combined["Other"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Other Christian"] = combined["Other Christian"].fillna((((combined["Other Christian"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Other Misc"] = combined["Other Misc"].fillna((((combined["Other Misc"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Pentecostal / Charismatic"] = combined["Pentecostal / Charismatic"].fillna((((combined["Pentecostal / Charismatic"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Protestant Denomination"] = combined["Protestant Denomination"].fillna((((combined["Protestant Denomination"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    combined["Zoroastrian"] = combined["Zoroastrian"].fillna((((combined["Zoroastrian"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(int)
    
    combined["Area in square miles - Total area"] = combined["Area in square miles - Total area"].fillna(combined["Area in square miles - Total area"].mean())
    combined["Area in square miles - Water area"] = combined["Area in square miles - Water area"].fillna(combined["Area in square miles - Water area"].mean())
    combined["Area in square miles - Land area"] = combined["Area in square miles - Land area"].fillna(combined["Area in square miles - Land area"].mean())
    
    combined["WHITE_rate"] = combined["WHITE_rate"].fillna((((combined["WHITE_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)
    combined["BLACK_rate"] = combined["BLACK_rate"].fillna((((combined["BLACK_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)
    combined["NATIVE_AMERICAN_rate"] = combined["NATIVE_AMERICAN_rate"].fillna((((combined["NATIVE_AMERICAN_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)
    combined["HAWAIIAN_PACIFIC_rate"] = combined["HAWAIIAN_PACIFIC_rate"].fillna((((combined["HAWAIIAN_PACIFIC_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)
    combined["MULTI_rate"] = combined["MULTI_rate"].fillna((((combined["MULTI_rate"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)
    combined["average_age"] = combined["average_age"].fillna((((combined["average_age"] * combined["POP_ESTIMATE_2015"]).sum()) / (combined["POP_ESTIMATE_2015"].sum()))).astype(float)
    
    combined["voter_turnout_rate"] = combined["votes"] / combined["POP_ESTIMATE_2015"]
    combined["Democrat"] = np.where(combined["D"] > combined["R"], 1, 0)
    combined["Republican"] = np.where(combined["R"] > combined["D"], 1, 0)
    combined["party_winner"] = np.where(combined["R"] > combined["D"], "Repulican", "Democrat")
    
    #drop unnecessary columns
    combined.drop('1', axis=1, inplace=True)
    combined.drop('2', axis=1, inplace=True)
    combined.drop('3', axis=1, inplace=True)
    combined.drop('4', axis=1, inplace=True)
    combined.drop('5', axis=1, inplace=True)
    combined.drop('6', axis=1, inplace=True)
    combined.drop('7', axis=1, inplace=True)
    combined.drop('8', axis=1, inplace=True)
    combined.drop('9', axis=1, inplace=True)
    combined.drop('10', axis=1, inplace=True)
    combined.drop('11', axis=1, inplace=True)
    combined.drop('12', axis=1, inplace=True)
    combined.drop('13', axis=1, inplace=True)
    combined.drop('14', axis=1, inplace=True)
    combined.drop('15', axis=1, inplace=True)
    combined.drop('16', axis=1, inplace=True)
    combined.drop('17', axis=1, inplace=True)
    combined.drop('18', axis=1, inplace=True)
    
    combined.drop('TOT_POP', axis=1, inplace=True)
    combined.drop('TOT_MALE', axis=1, inplace=True)
    combined.drop('TOT_FEMALE', axis=1, inplace=True)
    combined.drop('WA_MALE', axis=1, inplace=True)
    combined.drop('WA_FEMALE', axis=1, inplace=True)
    combined.drop('BA_MALE', axis=1, inplace=True)
    combined.drop('BA_FEMALE', axis=1, inplace=True)
    combined.drop('IA_MALE', axis=1, inplace=True)
    combined.drop('IA_FEMALE', axis=1, inplace=True)
    combined.drop('AA_MALE', axis=1, inplace=True)
    combined.drop('AA_FEMALE', axis=1, inplace=True)
    combined.drop('NA_MALE', axis=1, inplace=True)
    combined.drop('NA_FEMALE', axis=1, inplace=True)
    combined.drop('TOM_MALE', axis=1, inplace=True)
    combined.drop('TOM_FEMALE', axis=1, inplace=True)
    
    
    #output final combined csv
    combined.to_csv('combined_data.csv')
    

def clean_and_combine():
    download_census_files()
    csv_from_xls()
    income_clean()
    education_clean()
    population_clean()
    unemployment_clean()
    combine_files()

clean_and_combine()





