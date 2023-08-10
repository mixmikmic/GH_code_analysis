#Import modules:
import pandas as pd
import numpy as np
import scipy as sp
import sklearn

#import data and merge the macro onto the train and test
trainsm_df = pd.read_csv("train_small.csv", parse_dates=['timestamp'])
testsm_df = pd.read_csv("test_small.csv", parse_dates=['timestamp'])
macrosm_df = pd.read_csv("macro_small.csv", parse_dates=['timestamp'])
trainsm_df = pd.merge(trainsm_df, macrosm_df, how='left', on='timestamp')
testsm_df = pd.merge(testsm_df, macrosm_df, how='left', on='timestamp')
print(trainsm_df.shape, testsm_df.shape)

#attach the okrug region
okurg_df = pd.read_csv("okurg.csv")
trainsm_df = pd.merge(trainsm_df, okurg_df, how='left', on='sub_area')
testsm_df = pd.merge(testsm_df, okurg_df, how='left', on='sub_area')

#import chase data
chase_imputed = pd.read_csv("chase_imputed.csv")

chase_imputed.describe()

trainsm_merge = trainsm_df.drop(["full_sq", "kitch_sq", "floor", "max_floor"], axis=1)

train_chase = pd.merge(chase_imputed, trainsm_merge, how='left', on='id')

train_chase.describe()

missing = train_chase.isnull().sum()
print(missing.to_string())
print len(train_chase)

train_clean = train_chase.dropna(subset = ['metro_min_walk'])

missing = testsm_df.isnull().sum()
print(missing.to_string())

testsm_df.describe() #need to fix up kitch_sq, full_sq, max_floor

testsm_df.loc[testsm_df['max_floor'] < testsm_df['floor']] #need to fix 643 out of 7662
#notice that the kitchen square is the same as the max floor for these

643.0/7662.0 #8% of the test data has messed up kitchen, max floor

testsm_df.loc[testsm_df['floor'] < 1] #floor doesn't seem to be messed up... 

testsm_df.loc[testsm_df['full_sq'] < testsm_df['life_sq']] #fix full square, get rid of life square 

testsm_df.loc[testsm_df['full_sq'] < 1] #fix full square, get rid of life square 

testsm_df.loc[testsm_df['full_sq'] > 300] #fix full square, get rid of life square





test_clean = testsm_df.dropna(subset = ['metro_min_walk'])

train_clean = train_clean.dropna(axis=1, how='any')

test_clean = test_clean.dropna(axis=1, how='any')
test_clean = test_clean.drop(["material", "num_room"], axis=1)

train_clean.describe()

test_clean.describe()



#Population Density (will be the same throughout each SubArea)
train_clean["pop_density"] = train_clean["raion_popul"] / train_clean["area_m"].astype("float")
test_clean["pop_density"] = test_clean["raion_popul"] / test_clean["area_m"].astype("float")



#Ratio of elder population (will be the same throughout each SubArea)
train_clean["elder_ratio"] = train_clean["ekder_all"] / (train_clean["young_all"] + train_clean["work_all"] + train_clean["ekder_all"]).astype("float")
test_clean["elder_ratio"] = test_clean["ekder_all"] / (test_clean["young_all"] + test_clean["work_all"] + test_clean["ekder_all"]).astype("float")

#Ratio of under 18 population (will be the same throughout each SubArea)
train_clean["youth_ratio"] = train_clean["young_all"] / (train_clean["young_all"] + train_clean["work_all"] + train_clean["ekder_all"]).astype("float")
test_clean["youth_ratio"] = test_clean["young_all"] / (test_clean["young_all"] + test_clean["work_all"] + test_clean["ekder_all"]).astype("float")



test_clean.describe()

train_clean.describe()



#do a percent of floor value for train, and need to FIX TEST--how?

#Ratio of floor (unique to each unit)
train_clean["floor_ratio"] = train_clean["floor"] / train_clean["max_floor"].astype("float")

#Need to fix test
#test_clean["floor_ratio"] = test_clean["floor"] / test_clean["max_floor"].astype("float")


test_clean.loc[test_clean['max_floor'] < test_clean['floor']] #need to fix 640 out of 7628 



train_clean.describe()

#create year and month columns:

train_clean['year'] = pd.DatetimeIndex(train_clean['timestamp']).year
train_clean['month'] = pd.DatetimeIndex(train_clean['timestamp']).month

test_clean['year'] = pd.DatetimeIndex(test_clean['timestamp']).year
test_clean['month'] = pd.DatetimeIndex(test_clean['timestamp']).month





features = ['id',
 'timestamp',
 'full_sq',
 'floor',
 'max_floor',
 'kitch_sq',
 'product_type',
 'sub_area',
 'metro_min_walk',
 'kindergarten_km',
 'park_km',
 'kremlin_km',
 'oil_chemistry_km',
 'nuclear_reactor_km',
 'big_market_km',
 'market_shop_km',
 'detention_facility_km',
 'public_healthcare_km',
 'university_km',
 'workplaces_km',
 'preschool_km',
 'big_church_km',
 'okurg_district',
 'pop_density',
 'elder_ratio',
 'youth_ratio',
 'floor_ratio',
 'year',
 'month',
 'oil_urals',
 'cpi',
 'eurrub',
 'average_provision_of_build_contract_moscow',
 'micex',
 'mortgage_rate',
 'rent_price_4+room_bus',
 'sd_oil_yearly',
 'sd_cpi_yearly',
 'sd_eurrub_yearly',
 'sd_micex_yearly',
 'sd_mortgage_yearly',
 'sd_rent_yearly',
 'price_doc']

train_trial2 = train_clean[features]

train_trial2.describe()

#did not dummify/label encode the okurg_district, sub_area, product_type yet.

train_trial2

train_trial2.to_csv('trial_brandy2.csv', index = False)





#import data and merge the macro onto the train and test
testvsm_df = pd.read_csv("test_vsmall.csv", parse_dates=['timestamp'])
macrosm_df = pd.read_csv("macro_small.csv", parse_dates=['timestamp'])
okurg_df = pd.read_csv("okurg.csv")
chase_test = pd.read_csv("chase_final_test.csv")

testvsm_df.describe()
#chase_test.describe()

testvsm_df = pd.merge(testvsm_df, macrosm_df, how='left', on='timestamp')
testvsm_df = pd.merge(testvsm_df, okurg_df, how='left', on='sub_area')
test_chase = pd.merge(testvsm_df, chase_test, how='left', on='id')

test_chase.describe()

#pull out year and month
test_chase['year'] = pd.DatetimeIndex(test_chase['timestamp']).year
test_chase['month'] = pd.DatetimeIndex(test_chase['timestamp']).month

#floor to max floor ratio
test_chase["floor_ratio"] = test_chase["floor"] / test_chase["max_floor"].astype("float")

#Ratio of elder population (will be the same throughout each SubArea)
test_chase["elder_ratio"] = test_chase["ekder_all"] / (test_chase["young_all"] + test_chase["work_all"] + test_chase["ekder_all"]).astype("float")

#Ratio of under 18 population (will be the same throughout each SubArea)
test_chase["youth_ratio"] = test_chase["young_all"] / (test_chase["young_all"] + test_chase["work_all"] + test_chase["ekder_all"]).astype("float")

#Population Density (will be the same throughout each SubArea)
test_chase["pop_density"] = test_chase["raion_popul"] / test_chase["area_m"].astype("float")



features_test = ['id',
 'timestamp',
 'full_sq',
 'floor',
 'max_floor',
 'product_type',
 'sub_area',
 'metro_min_walk',
 'kindergarten_km',
 'park_km',
 'kremlin_km',
 'oil_chemistry_km',
 'nuclear_reactor_km',
 'big_market_km',
 'market_shop_km',
 'detention_facility_km',
 'public_healthcare_km',
 'university_km',
 'workplaces_km',
 'preschool_km',
 'big_church_km',
 'okurg_district',
 'pop_density',
 'elder_ratio',
 'youth_ratio',
 'floor_ratio',
 'year',
 'month',
 'oil_urals',
 'cpi',
 'eurrub',
 'average_provision_of_build_contract_moscow',
 'micex',
 'mortgage_rate',
 'rent_price_4+room_bus',
 'sd_oil_yearly',
 'sd_cpi_yearly',
 'sd_eurrub_yearly',
 'sd_micex_yearly',
 'sd_mortgage_yearly',
 'sd_rent_yearly']

test_final = test_chase[features_test]

test_final.describe()

test_final.to_csv('test_final.csv', index = False)

#did not dummify/label encode the okurg_district, sub_area, product_type yet.



train_1 = pd.read_csv("trial_brandy2.csv", parse_dates=['timestamp'])
train_2 = pd.read_csv("train_final.csv", parse_dates=['timestamp'])

train_1.describe()

train_2.describe()



