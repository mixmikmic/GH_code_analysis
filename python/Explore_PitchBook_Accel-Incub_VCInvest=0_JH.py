import pandas as pd
import datetime
import yaml
import os
import tweepy

acc_incub_general_info_df = pd.read_excel("../data/raw/CA_Accel-Incub-Seed_PitchBook/Company_General_Information.xlsx",header=6)
acc_incub_general_info_df.head(2)
# interesting columns= Company ID (primary key), Description, Company Name, HQ Post Code, 
#Primary Industry Code, Primary Contact, Year Founded, Active Investors

acc_incub_last_financing_df = pd.read_excel("../data/raw/CA_Accel-Incub-Seed_PitchBook/Last_Financing_Details.xlsx",header=6)
acc_incub_last_financing_df .head(2)
# interesting columns = Company ID ( primary key), Company Name, Growth Rate, Size Multiple, last financing date,
# last financing Size, Last financing valuation, Last Financing Deal Type 2

# Note : Only want series A or later, filter OUT the seed rounds

acc_incub_company_financials_df = pd.read_excel("../data/raw/CA_Accel-Incub-Seed_PitchBook/Public_Company_Financials.xlsx",header=6)
acc_incub_company_financials_df.head(2)
# Interesting columns are NOTHING

acc_incub_social_web_df =  pd.read_excel("../data/raw/CA_Accel-Incub-Seed_PitchBook/Social_and_Web_Presence.xlsx",header=6)
acc_incub_social_web_df.head(2)
# interesting columns = company id (primary key), company name, growth rate, size multiple, majestic referring domains
# facebook likes, Tiwtter followers, Employees, Total raised

acc_incub_general_info_colDrop_df = acc_incub_general_info_df[["Company ID", "Description", "Company Name", "HQ Post Code", "Primary Industry Code",
                                         "Primary Contact", "Year Founded", "Active Investors","HQ Location"]]
acc_incub_last_financing_colDrop_df =acc_incub_last_financing_df [["Company ID", "Growth Rate", "Size Multiple",
                                    "Last Financing Date","Last Financing Size","Last Financing Valuation",
                                                   "Last Financing Deal Type 2 "]]
acc_incub_social_web_colDrop_df  =acc_incub_social_web_df [["Company ID",  "Growth Rate", 
                                              "Size Multiple", "Majestic Referring Domains",
        "Facebook Likes", "Twitter Followers", "Employees", "Total Raised"]]
final_acc_incub_df = acc_incub_general_info_colDrop_df.merge(acc_incub_last_financing_colDrop_df, on='Company ID').merge(acc_incub_social_web_colDrop_df, 
                                                                                     on='Company ID')

final_acc_incub_df.info()

final_acc_incub_df .drop([
                  'Growth Rate_y','Size Multiple_y'],axis=1,inplace=True)
final_acc_incub_df.rename(columns={'Growth Rate_x':'Growth Rate',"Size Multiple_x":'Size Multiple',
                           "Company Name_x":"Company Name"},inplace=True) # rename cols



final_acc_incub_df['Last Financing Deal Type 2 '].unique()

final_acc_incub_filter_df= final_acc_incub_df.loc[(final_acc_incub_df['Last Financing Deal Type 2 ']!='Series A') & 
                (final_acc_incub_df['Last Financing Deal Type 2 ']!='Series AA') &
                   (final_acc_incub_df['Last Financing Deal Type 2 ']!='Series B') &
            (final_acc_incub_df['Last Financing Deal Type 2 ']!='Series A1') &
             (final_acc_incub_df['Last Financing Deal Type 2 ']!='Series 2') &
            (final_acc_incub_df['Last Financing Deal Type 2 ']!='Series A2') &
            (final_acc_incub_df['Last Financing Deal Type 2 ']!='Series B1') &
            (final_acc_incub_df['Last Financing Deal Type 2 ']!='Series C') &
            (final_acc_incub_df['Last Financing Deal Type 2 ']!='Series A3'),: ]

final_acc_incub_filter_df.info()

final_acc_incub_filter_df = final_acc_incub_filter_df.loc[
    (final_acc_incub_filter_df['HQ Post Code'].isnull()==False) &
    (final_acc_incub_filter_df['Year Founded'].isnull()==False) &
    (final_acc_incub_filter_df['Primary Contact'].isnull()==False) &
    (final_acc_incub_filter_df['Last Financing Date'].isnull()==False)  &
    (final_acc_incub_filter_df['Last Financing Size'].isnull()==False),: ]

final_acc_incub_filter_df.info()

final_acc_incub_filter_df.describe()

median_values={}
for row in final_acc_incub_filter_df.describe(): # get median values
    if row =='Last Financing Valuation': # don't have enough data for this
        pass
    else:
        median_values[row]=final_acc_incub_filter_df.describe()[row]["50%"]

median_values

imputed_final_df = final_acc_incub_filter_df.copy()
for key in median_values: # update the nan values with the median
    updated_col = final_acc_incub_filter_df.loc[:,key].copy()
    updated_col = updated_col.fillna(median_values[key])
    imputed_final_df.loc[:,key] = updated_col

imputed_final_df.info()

ca_income = pd.read_csv("../data/raw/median_income_tables/ca_income_by_county.csv")

ca_income.median_household_income[ca_income.median_household_income!="[7"].describe()

# For places without median income information, fill in with the median of CA

ca_income.median_household_income = ca_income.median_household_income.replace("[7","$42,500")

ca_income.median_household_income =     ca_income.median_household_income.apply( lambda x: int(x.strip("$").replace(",",""))) # conver to numbers

ca_income.median_household_income.describe()

date_pulled = datetime.datetime(2017,6,9)

imputed_final_df["days_since_offer"]=  imputed_final_df["Last Financing Date"].apply(
    lambda x: (date_pulled - x).days)

imputed_final_df["days_since_offer"].head()

imputed_final_df.info()

t = ca_income[ca_income.Place=="Mountain View"]['median_household_income'].values

vc_invest = []
for row in imputed_final_df.iterrows():
    row = row[1]
    place = row['HQ Location'].split(",")[0]
    state = row['HQ Location'].split(",")[1]
    try:
        # median year income
        median_yr_income = int(ca_income[ca_income.Place==place]['median_household_income'].values) 
        daily_income = median_yr_income/365
   
        daily_income_for_company = row['Employees'] * daily_income
        #print(row['Employees'],"employees")
        #print(daily_income_for_company ,'daily_income_for_company ')

        required_funding_total = daily_income_for_company * (row['days_since_offer'])
        #print(required_funding_total,'required_funding_total')
        # funding so far in dollars (from last fundraising round)
        funding_so_far = row['Last Financing Size']*1_000_000
        #print(funding_so_far,'funding_so_far')
        #print()
        if required_funding_total > funding_so_far:
            vc_invest.append(0)
        else:
            vc_invest.append('NaN')
    except Exception as e:
        if state == "CA":
            # median year income
            median_yr_income = int(ca_income[ca_income.Place==place]['median_household_income'].values[1]) 
            daily_income = median_yr_income/365
            daily_income_for_company = row['Employees'] * daily_income
            required_funding_total = daily_income_for_company *row['days_since_offer']
            funding_so_far = row['Last Financing Size']*1_000_000
            if required_funding_total < funding_so_far:
                vc_invest.append(0)
            else:
                vc_invest.append('NaN')
        else:
            vc_invest.append('NaN')

imputed_final_df['vc_invest']=vc_invest

imputed_final_df= imputed_final_df[imputed_final_df['vc_invest']!='NaN']

imputed_final_df.info()

def username_search(name, company, state, c = 20):
    """Run a search on twitter for the given name. Returns the first username (should be the most relevant).
    Looks to match a state location with the state locatio nof the company

    First try searching for the person's name + company. If that does not work, try just searching for the 
    person's name.
    
    Count is for the number of results to return. Default to twenty. If not in the first twenty results, probably not
    the correct user"""
    state = state.lower()
    
    credentials = yaml.load(open(os.path.expanduser('~/.ssh/api_credentials.yml')))
    auth = tweepy.OAuthHandler(credentials['twitter']['consumer_key'], credentials['twitter']['consumer_secret'],)
    auth.set_access_token(credentials['twitter']['token'], credentials['twitter']['token_secret'])
    api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    

    try: # search the name and the company
        tweets = api.search_users(q=str(name)+" "+str(company), count=c)
        test = result[0].screen_name
        screen_n = None
        
        for result in tweets:
            location = result.location.lower().split(" ") # see if the location is in the companies state
            if state in location:
                return result.screen_name

        if screen_n == None:
            return 'NaN'
        else:
            return screen_n
            
    except Exception as e: # try just the name
        try:

            tweets = api.search_users(q=name, count = c)
            
            screen_n = None
            for result in tweets:
                if state in result.location.lower().split(" "):

                    return result.screen_name
            if screen_n == None:
                return "NaN"
            else:
                return screen_n
            

        except Exception as e:
            return "NaN"

twitter_usernames_accel_incub_df = []
for idx,row in enumerate(imputed_final_df.iterrows()):
    location = row[1]['HQ Location'].split(",")[1].strip(" ")
    company = row[1]['Company Name']
    founder = row[1]['Primary Contact']

    twitter_usernames_accel_incub_df.append(username_search(founder,company, location ))
    if idx%100 ==0:
        print(f"Finished {idx/len(imputed_final_df)}")

imputed_final_df['Twitter_Username'] = twitter_usernames_accel_incub_df

finished_acc_incub_df =  imputed_final_df[
        (imputed_final_df.Twitter_Username!='NaN') ]

finished_acc_incub_df .info()

finished_acc_incub_df.to_csv("../data/processed/PitchBook_CA_VCInvest=0.csv")



