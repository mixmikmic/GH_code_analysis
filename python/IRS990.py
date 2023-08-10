import csv
import random
import xml.etree.ElementTree as ET
import requests

def sampling(size, f_name):
    """
    Random sampling `size` samples using the f_name csv file as population. 
    Reservoir sampling algorithm is used.

    Args:
        size(int) : sample size
        f_name(string) : name of the population file

    Returns:
        int list : samples, each element is the unique identifier of the filing
    """
    samples = []
    counter = 0
    with open(f_name, 'r') as fp:
        first_line = fp.readline()
        for line in fp:
            counter += 1
            # Fill in the samples 
            if len(samples) < size:
                samples.append(line.strip('\n')[-18:])
            # Dynamic probability of replacing samples with the new sample
            else:
                indicator = int(random.random() * counter)
                # With size/counter probability
                if indicator < size:
                    samples[indicator] = line.strip('\n')[-18:]
    return samples


def get_data(samples, output, *interests):
    """
    Accessing each sample, organize and write the interested entires.

    Args:
        samples(str list) : list of unique identifiers of the samples
        output(str) : file name of the output
        *interests : multiple string of the tags of interested entries. If the
            tags are not in the 990 form, values will be replaced with empty string

    """
    
    with open(output, 'w') as out_csv:
        # The headers are the identifier and interested tags
        writer = csv.DictWriter(out_csv, fieldnames = ['id'] + list(interests), 
                                delimiter = '\t')
        writer.writeheader()

        url_p = 'https://s3.amazonaws.com/irs-form-990/'
        url_e = '_public.xml'
        
        # Use counter to track the task rate
        counter = 0
        total = float(len(samples))

        for sample in samples:
            counter += 1
            print("Finished " + str(counter / total * 100) + "%")
            # Parse the xml file
            xml_response = requests.get(url_p + sample + url_e)
            root = ET.fromstring(xml_response.content)

            # Get the data for interested tags
            data = {'id' : sample}

            for tag in interests:
                # For missing data, we use empty string as replacement
                try:
                    data[tag] = next(root.iter('{http://www.irs.gov/efile}' + 
                                               tag)).text
                except StopIteration:
                    data[tag] = ''

            writer.writerow(data)

import sample

curr_sample = sample.sampling(30000, 'index_2015.csv')
sample.get_data(curr_sample, 'sample_2015.csv', 
                'PYContributionsGrantsAmt', 
                'CYContributionsGrantsAmt', 
                'TotalContributionsAmt', 
                'ContributionsGiftsGrantsEtcAmt',
                'PYInvestmentIncomeAmt', 
                'CYInvestmentIncomeAmt')

with open('sample_2015.csv', 'r') as fp:
    length = 0
    for line in fp:
        # Illustrate one part of the csv file
        if length < 15:
            print(line)
        length += 1

print(length)

with open('sample_2015.csv', 'r') as fp:
    with open('interest.csv', 'w') as out_fp:
        # Header line
        out_fp.write(fp.readline())
        for line in fp:
            entries = line.split('\t')
            
            # We only want the data with  valid PYContributionsGrantsAmt 
            # and PYInvestmentIncomeAmt
            if entries[1] and entries[1] != "RESTRICTED" and             entries[5] and entries[5] != "RESTRICTED":
                out_fp.write(line)

# Ilustrate the final intesrest.csv (First 15 lines)
with open('interest.csv', 'r') as fp:
    length = 0
    for line in fp:
        if length < 15:
            print(line)
        length += 1

print(length)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants for the key
CONTRI = "PYContributionsGrantsAmt"
INVEST = "PYInvestmentIncomeAmt"

# Build data frame for analysis, from_csv() reader would maek some errors while parsing
df = pd.DataFrame.from_csv('interest.csv', sep = '\t')

# Vertualizing the Dataframe (first 15 rows)
# The output is a little odd in jupyter
print(df[:16])

print(df[CONTRI].mean())
print(df[CONTRI].median())
print(df[CONTRI].std())

print(df[INVEST].mean())
print(df[INVEST].median())
print(df[INVEST].std())

get_ipython().magic('matplotlib inline')
df.boxplot(CONTRI)

df.boxplot(INVEST)

sca = df.plot(x=INVEST, y=CONTRI, style='o')
sca.set_ylim(0,10000000)

