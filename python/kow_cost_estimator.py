armies = ['Undead Armies', 'Goblins']

# create a list of df from the army list above
# use pd.concat method to create final df per pandas documentation
import pandas as pd

units_list = []
for army in armies:
    units_list.append(pd.read_csv(army + '.csv', encoding="utf-8-sig"))

units = pd.concat(units_list)
units.reset_index(inplace=True, drop=True)
units

import re

class MLRTransform:
    """Transforms raw data into Multiple Linear Regression ready dataframe"""
    #ToDo: turn this class into a child of pd.DataFrame instead of standalone
    def __init__(self, df):
        self.raw_data = df
          
    def transform(self):
        method_man = [self.__army_name(), self.__army_allegiance(), self.__unit_name(),
                     self.__unit_type(), self.__unit_size(), self.__sp(),
                     self.__me(), self.__ra(), self.__de(),
                     self.__att(), self.__ne(), self.__special()]
        
        x = pd.concat(method_man, axis=1)
        y = self.__points()
        return x, y
    
    def __army_name(self):
        return self.__transform_column('Army Name')
    
    def __army_allegiance(self):
        return self.__transform_column(' Army Allegiance')
    
    def __unit_name(self):
        #include individuals and irregulars
        df = pd.DataFrame()
        df['Unique'] = ""
        df['Irregular'] = ""
        for row in self.raw_data[' Unit Name']:
            if row.endswith('[1]'):
                i = {'Unique': 1.0,
                    'Irregular': 0.0}
                df = df.append(i, ignore_index=True)
            elif row.endswith('*'):
                i = {'Unique': 0.0,
                    'Irregular': 1.0}
                df = df.append(i, ignore_index=True)
            else:
                i = {'Unique': 0.0,
                    'Irregular': 0.0}
                df = df.append(i, ignore_index=True)
        
        df.fillna(0, inplace=True)
        return df
    
    def __unit_type(self):
        return self.__transform_column(' Unit Type')
    
    def __unit_size(self):
        return self.__transform_column(' Unit Size')
    
    def __sp(self):
        return pd.to_numeric(self.raw_data[' Sp'], downcast='float')
    
    def __me(self):
        return pd.to_numeric(self.raw_data[' Me'], downcast='float')
    
    def __ra(self):
        df = pd.to_numeric(self.raw_data[' Ra'], downcast='float')
        values = [i for i in range(7, 1, -1)]
        
        for i, v in enumerate(df):
            if v == 0.0:
                pass
            else:
                df[i] = values[int(v)]
            
        return df
    
    def __de(self):
        return pd.to_numeric(self.raw_data[' De'], downcast='float')
        
    def __att(self):
        return pd.to_numeric(self.raw_data[' Att'], downcast='float')
    
    def __ne(self):
        # iterate, divide at '/', and turn into waver and route columns
        # turn 0 values into new column; 1 if fearless, else 0
        df = pd.DataFrame()
        columns = ['Fearless', 'NeW', 'NeR']
        for col in columns:
            df[col] = ""
            
        for row in self.raw_data[' Ne']:
            new, ner = row.split('/')
            new = new[1::]
            new = float(new)
            ner = float(ner)
            if new == 0.0:
                i = {'Fearless': 1.0,
                    'NeW': 0.0,
                    'NeR': ner}
                df = df.append(i, ignore_index=True)
            else:
                i = {'Fearless': 0.0,
                    'NeW': new,
                    'NeR': ner}
                df = df.append(i, ignore_index=True)
                    
        df.fillna(0, inplace=True)
        return df
    
    def __points(self):
        return pd.to_numeric(self.raw_data[' Pts'], downcast='float')
    
    def __special(self):
        df = pd.DataFrame()
        unique_values = []
        for row in self.raw_data[' Special']:
            values = row.split(';')
            for value in values:
                if value in unique_values:
                    pass
                else:
                    unique_values.append(value)
                    
        new_unique_values = []            
        for v in unique_values:
            new_v = re.sub(r"\(.*\)","", v)
            if new_v in new_unique_values:
                pass
            else:
                new_unique_values.append(new_v)
            
        for value in new_unique_values:
            df[value] = ""
            
        for row in self.raw_data[' Special']:
            values = row.split(';')
            i = {}
            for value in values:
                rec = re.compile("\d")
                digit = rec.findall(value)
                new_v = re.sub(r"\(.*\)","", value)
                i[new_v] = 1.0
                reference_index = [i for i in range(7, 1, -1)]
                if digit:
                    scalar = ""
                    for d in digit:
                        scalar = scalar + d
                    if ' Regeneration' in value:
                        i[new_v] = reference_index[int(d)]
                    else:
                        i[new_v] *= float(scalar)
            df = df.append(i, ignore_index=True)                    
            
        df.fillna(0, inplace=True)
        return df
            
    def __transform_column(self, column_name):
        unique_values = self.raw_data[column_name].unique()
        df = pd.DataFrame()
        
        # loop should create a new column for each unique value
        for value in unique_values:
            df[value] = ""
            
        # loop should iterate over each row in raw_data[column_name] and create a row in df with a 1 in the column 
        # that matches its value
        for row in self.raw_data[column_name]:
            i = {row: 1.0}
            df = df.append(i, ignore_index=True)
            
        df.fillna(0, inplace=True)
                    
        return df
    
equation_df = MLRTransform(units)
x, y = equation_df.transform()
x.describe()

# take a look at the columns to ensure nothing looks odd
col = x.columns
print(col)

# take a look at outputs
y.describe()

# create a linear model for the data
from sklearn import linear_model

lm = linear_model.LinearRegression()
model = lm.fit(x, y)

lm.score(x, y)

# placeholder work until a proper import function is developed
# xt represents the stats for a given unit
xt = [1, 0, 1, 0, 0, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 5,
    4, 0, 5, 20, 1, 0, 25, 1,
    2, 0, 0, 0,
    0, 0, 5, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0]

coeff = lm.coef_
value = 0
for i, v in enumerate(xt):
    value += v*coeff[i]

value += lm.intercept_

# output point value of xt
print(value)

