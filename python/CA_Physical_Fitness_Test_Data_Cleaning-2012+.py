#Import Packages
import os
import pandas as pd


#Set file path containing data
datapath = "/Users/nwchen24/Desktop/UC_Berkeley/w209_Data_Viz/final_project_data"

#initialize lists to hold filepaths
Phys_files_list = []
Entities_files_list = []

#Walk the data directory and get all filepaths
for root, dirs, files in os.walk(datapath):
    for filename in files:
        #Get full list of filepaths to the physical fitness test files
        if filename.endswith('.txt'):    
            if filename[:4] == "Phys":
                 Phys_files_list.append(datapath + "/PFT_" + filename[7:11] + "/" + filename)
            if filename[8:16] == "Research":
                Phys_files_list.append(datapath + "/PFT_" + str(int(filename[:4])+1) + "/" + filename)

            #Get full list of filepaths to the entities files        
            if filename[:8] == "Entities":
                Entities_files_list.append(datapath + "/PFT_" + filename[8:13] + "/" + filename)
            if filename[8:16] == "Entities":
                Entities_files_list.append(datapath + "/PFT_" + str(int(filename[:4])+1) + "/" + filename)

#get list of all columns in the file from each year
Phys_col_list = []

#read PhysFit files
for filepath in Phys_files_list:
    
    if int(filepath[73:77]) < 2011:
        pass
    
    #Files starting in 2012 are tab delimited
    elif (int(filepath[73:77]) < 2012) | (int(filepath[73:77]) > 2013):
        #read the file
        temp_df = pd.read_csv(filepath)
        #print the shape
        print filepath[73:77]
        print temp_df.shape
        #get the columns
        temp_col_list = temp_df.columns
        #add columns not already encountered to the column list
        for colname in temp_col_list:
            if colname not in Phys_col_list:
                Phys_col_list.append(colname)
    else:
        #read the file
        temp_df = pd.read_table(filepath)
        #print the shape
        print filepath[73:77]
        print temp_df.shape
        #get the columns
        temp_col_list = temp_df.columns
        #add columns not already encountered to the column list
        for colname in temp_col_list:
            if colname not in Phys_col_list:
                Phys_col_list.append(colname)

    

#get list of all columns in the file from each year
test_list = []

#read PhysFit files
for filepath in Phys_files_list:
    
    #Files starting in 2012 are tab delimited
    if (int(filepath[73:77]) < 2012) | (int(filepath[73:77]) > 2013):
        #read the file
        temp_df = pd.read_csv(filepath)
        temp_col_list = temp_df.columns
        
    else:
        #read the file
        temp_df = pd.read_table(filepath)
        temp_col_list = temp_df.columns
        if 'ID' in temp_col_list:
            print filepath
        if 'Line_Text' in temp_col_list:
            print filepath

    

#Create a dict that maps all possible column names to standardized column names
#Standardized values come second
Physfit_column_mapping = {}
Physfit_column_mapping['Level'] = 'Aggregation_Lvl'
Physfit_column_mapping['SubGrp'] = 'SubGrp'
Physfit_column_mapping['RptType'] = 'RptType'
Physfit_column_mapping['Line_num'] = 'Line_num'
Physfit_column_mapping['line_num'] = 'Line_num'
Physfit_column_mapping['Line_Number'] = 'Line_num'
Physfit_column_mapping['Table_Number'] = 'Table_Number'
Physfit_column_mapping['Report_Number'] = 'Report_Number'
Physfit_column_mapping['Line_Text'] = 'Test_Description'
Physfit_column_mapping['ID'] = 'Record_ID_2011_Only'

Physfit_column_mapping['Ccode'] = 'CountyCode'
Physfit_column_mapping['CO'] = 'CountyCode'
Physfit_column_mapping['Dcode'] = 'DistrictCode'
Physfit_column_mapping['DIST'] = 'DistrictCode'
Physfit_column_mapping['charternum'] = 'CharterNum'
Physfit_column_mapping['ChrtNum'] = 'CharterNum'
Physfit_column_mapping['Scode'] = 'SchoolCode'
Physfit_column_mapping['SCHL'] = 'SchoolCode'
Physfit_column_mapping['cds_code'] = 'CountyDistSchCode'

Physfit_column_mapping['Gr05_Stu'] = 'Gr5_NumStu'
Physfit_column_mapping['Gr5PctIn'] = 'Gr5_PctPass'
Physfit_column_mapping['Gr5PctNotIn'] = 'Gr5_PctFail'

Physfit_column_mapping['Gr07_Stu'] = 'Gr7_NumStu'
Physfit_column_mapping['Gr7PctIn'] = 'Gr7_PctPass'
Physfit_column_mapping['Gr7PctNotIn'] = 'Gr7_PctFail'

Physfit_column_mapping['Gr09_Stu'] = 'Gr9_NumStu'
Physfit_column_mapping['Gr9PctIn'] = 'Gr9_PctPass'
Physfit_column_mapping['Gr9PctNotIn'] = 'Gr9_PctFail'

Physfit_column_mapping['NoStud5'] = 'Gr5_NumStu'
Physfit_column_mapping['NoHFZ5'] = 'Gr5_NumPass'
Physfit_column_mapping['Perc5a'] = 'Gr5_PctPass'
Physfit_column_mapping['Perc5b'] = 'Gr5_PctFail_Need_Improvement'
Physfit_column_mapping['Perc5c'] = 'Gr5_PctFail_High_Risk'

Physfit_column_mapping['NoStud7'] = 'Gr7_NumStu'
Physfit_column_mapping['NoHFZ7'] = 'Gr7_NumPass'
Physfit_column_mapping['Perc7a'] = 'Gr7_PctPass'
Physfit_column_mapping['Perc7b'] = 'Gr7_PctFail_Need_Improvement'
Physfit_column_mapping['Perc7c'] = 'Gr7_PctFail_High_Risk'

Physfit_column_mapping['NoStud9'] = 'Gr9_NumStu'
Physfit_column_mapping['NoHFZ9'] = 'Gr9_NumPass'
Physfit_column_mapping['Perc9a'] = 'Gr9_PctPass'
Physfit_column_mapping['Perc9b'] = 'Gr9_PctFail_Need_Improvement'
Physfit_column_mapping['Perc9c'] = 'Gr9_PctFail_High_Risk'

#Get list of the unique standardized column names to initialize DF
unique_Physfit_standard_cols = list(set( val for val in Physfit_column_mapping.values()))



#initialize dataframe to hold the physical fitness files
Physfit_df = pd.DataFrame(columns = unique_Physfit_standard_cols)

#read PhysFit files
for filepath in Phys_files_list:
    temp_df = pd.read_csv(filepath)
    temp_df = temp_df.rename(columns = Physfit_column_mapping)
    temp_df['Year'] = filepath[73:77]
    Physfit_df = Physfit_df.append(temp_df)
    print filepath[73:77] + " Read Successfully"
    

#Save serialized version to file
Physfit_df.to_pickle('/Users/nwchen24/Desktop/UC_Berkeley/w209_Data_Viz/final_project_data/Combined_Data/Combined_Physical_Fitness_Data.pkl')





