import numpy as np
import pandas as pd
from pandas import DataFrame
from Bio import Entrez

def GRID_to_PMID(GRID):
    '''use GRID to search PMID, return dict(PMID:GRID)'''
    Entrez.email = 'yier.jin@gmail.com'
    handle = Entrez.esearch(db="pubmed", term=GRID)
    record = Entrez.read(handle)
    handle.close()
    IdList = record["IdList"]
    GRID = str(GRID)
    dic = {}
    for i in IdList:
        dic[i] = GRID
    return (dic)

def GRID_list_to_PMID(GRID_list):
    """used a GRID list to search PMID, return dict(PMID:GRID)"""
    dic = {}
    for i in GRID_list:
        dic_i = GRID_to_PMID(i)
        dic.update(dic_i)
    return (dic)

def Extract_GRID_list (NIH_file):
    """Extract GRID list from NIH summary csv files, return a GRID list""" 
    df = pd.read_csv(NIH_file, encoding = "ISO-8859-1")
    GRID = df.iloc[:,0].dropna()
    GRID_list = GRID.tolist()
    return(GRID_list)

def NIH_GRID_to_PMID(GRID_list):
    """use estracted GRID, call "GRID_list_to_PMID" func to return PMID list,
       write into csv file."""
    GR_Dic = GRID_list_to_PMID(GRID_list)
    df = DataFrame()
    df["PMID"] = GR_Dic.keys()
    df["GRID"] = GR_Dic.values()
    print(df.head())
    df.to_csv("GrantID.to_PMID.txt")

NIH_file = "FY_2010_split1.txt"
GRID_list = Extract_GRID_list(NIH_file)
NIH_GRID_to_PMID(GRID_list)

def PMID_to_summary (PMID):
    """use PMID to search paper summary"""
    handle = Entrez.esummary(db="pubmed", id=PMID)
    record = Entrez.read(handle)
    handle.close()
    dic = {}
    Id = record[0]["Id"]
    PubDate = record[0]["PubDate"]
    PubTypeList = record[0]["PubTypeList"]
    FullJournalName = record[0]["FullJournalName"]
    dic[Id] = [PubDate, PubTypeList, FullJournalName]
    return dic

def PMID_list_to_summary (PMID_list):
    """use PMID list to search paper abstract"""
    dic = {}
    for i in PMID_list:
        dic_i = PMID_to_summary(i)
        dic.update(dic_i)
    return (dic)

import pandas as pd
file = "FY2010_merge_all.csv"
FY2010 = pd.read_csv(file)
FY2010 = FY2010.drop("Unnamed: 0", axis=1)
FY2010.head(6)

file = "Abstract.csv"
FY2010 = pd.read_csv(file)
FY2010 = FY2010.drop("Unnamed: 0", axis=1)
FY2010.head(6)

FY2010["PMAbs"][0]

FY2010["GRAbs"][0]



