import scipy.stats
import numpy as np
from random import randint
import pandas as pd
import math
import random 
random.seed(123) 

"""
Part 0:
create the customer information table(customer_id, gender):
"""
def customer_id_gender(p,num_customer):
    customer_gender = scipy.stats.bernoulli.rvs(p, size=num_customer)
    customer_id = list(range(num_customer))
    id_gender = np.column_stack((customer_id,customer_gender))
    customer_id_gender_df = pd.DataFrame(id_gender)
    customer_id_gender_df.columns = ["Customer_id","Gender"]
    return customer_id_gender_df
customer_id_gender_df = customer_id_gender(0.5, 1000)

print("The customer information table: ")
customer_id_gender_df[0:5]

"""
Part 1: 
create the customer information table: (customer_information_df)
Parameters: num_customer, p , session_range
Customer ~ uniform(num_customer)
Gender ~ bernoulli(p)
session_length ~ uniform(session_range)
"""
def simulate_customer_information(p,num_customer,session_range):
        
    session_length = []
    for i in range(num_customer):
        x = np.random.randint(1,session_range+1)
        session_length.append(x)

    customer_id = []
    customer_gender = []
    for i in range(num_customer):
        x = np.random.randint(0,num_customer)
        y = customer_id_gender_df.loc[x][1]
        customer_id.append(x)
        customer_gender.append(y)

    customer_information = np.column_stack((customer_id,customer_gender,session_length))
    customer_information_df = pd.DataFrame(customer_information)
    customer_information_df.columns = ["Customer","Gender","session_length"]
    
    return customer_information_df

p = 0.5
num_customer = 1000
session_range = 100
customer_information_df = simulate_customer_information(p,num_customer,session_range)

print("The session information table: ")
customer_information_df.loc[0:5]

"""
Part 2: 
create product information table: (product_information_df)
Parameters:
num_product
product_price ~ Beta(a,b)*(price for category 0,1,2: p_0,p_1,p_2)
"""
def simulate_product_information(num_product,a,b,p_0,p_1,p_2):
    
    category = []
    for i in range(num_product):
        x = np.random.randint(0,3)
        category.append(x)
    category_0 = category.count(0)
    category_1 = category.count(1)
    category_2 = category.count(2)

    list0 = list(np.zeros(category_0))
    list1 = list(np.zeros(category_1)+1)
    list2 = list(np.zeros(category_2)+2)
    category_list = list0+list1+list2

    product_0_price = scipy.stats.beta.rvs(a, b, size=category_0)*p_0
    product_1_price = scipy.stats.beta.rvs(a, b, size=category_1)*p_1
    product_2_price = scipy.stats.beta.rvs(a, b, size=category_2)*p_2

    product_id = [int(i) for i in range(num_product)]
    prices = list(product_0_price)+list(product_1_price)+list(product_2_price)

    product_information = np.column_stack((product_id,category_list,prices))
    product_information_df = pd.DataFrame(product_information, dtype='float')
    product_information_df.columns = ["product_id","category","prices"]
    product_information_df["product_id"] = product_information_df["product_id"].astype("int")
    product_information_df["category"] = product_information_df["category"].astype("int")
    
    return product_information_df

a = 2
b = 5
p_0 = 200
p_1 = 400
p_2 = 8000
num_product = 1000
product_information_df = simulate_product_information(num_product,a,b,p_0,p_1,p_2)

print("The product information table: ")
product_information_df.loc[0:5]

"""
Part 3: 
create shopping table for each customer: (shopping_df)
which contains each product id for each customer at one session 
customer_information_df = 
simulate_customer_information(p,num_customer,session_range)
"""
def simulate_shopping_information(num_customer,num_product,customer_information_df):
    shopping_product_id = []
    for i in range(num_customer):
        list1 = []
        for item in range(customer_information_df['session_length'][i]):
            x = np.random.randint(0,num_product)
            list1.append(x)
        shopping_product_id.append(list1)
    
    shopping_product_id_df = pd.DataFrame(shopping_product_id,dtype='int')  
    shopping_df = pd.concat([customer_information_df,shopping_product_id_df], axis=1)

    name = ["View_"+str(x) for x in range(100)]
    column_name = ["customer_id","customer_gender","session_length"]+name
    shopping_df.columns = column_name
    shopping_df = shopping_df.fillna("")
    
    return shopping_df

shopping_df = simulate_shopping_information(num_customer,num_product,customer_information_df)

shopping_df.loc[0:5]

"""
Part 4: (total_prob_df) and (total_0_1_df)
calculate the each seesion: buying or not?
p(buy|product,customer) = p(buy|customer_gender,product_category, product_price)
log(p/(1-p))=alpha_0 + alpha_11*Indicator{male}*{category_1} +
  alpha_12*Indicator{male}*{category_2}+alpha_13*Indicator{male}*{category_3}
  +[alpha_41+alpha_21*price+alpha_31*price^2]*{category_1}
  +[alpha_42+alpha_22*price+alpha_32*price^2]*{category_2}
  +[alpha_43+alpha_23*price+alpha_33*price^2]*{category_3}  
  alpha_0 = log(0.1/(1-0.1))
  alpha_11 = 1; alpha_12 = -1; alpha_13 = -1
"""
# define logistic regression model: x~price 
#lr_m0,lr_m1,lr_m2: for male and product category 0,1,2
#lr_f0,lr_f1,lr_f2: for female and product category 0,1,2
def lr_m0(x):
    return math.log(1/9) -1 + (-10.0/(100**2)*(x-100)**2+10.0)
def lr_m1(x):
    return math.log(1/9) -1 + (-10.0/(200**2)*(x-200)**2+10.0)
def lr_m2(x):
    return math.log(1/9) +1 + (-1.0/(4000**2)*(x-4000)**2+1.0)
def lr_f0(x):
    return math.log(1/9) +1 + (-10.0/(100**2)*(x-100)**2+10.0)
def lr_f1(x):
    return math.log(1/9) +1 + (-10.0/(200**2)*(x-200)**2+10.0)
def lr_f2(x):
    return math.log(1/9) -1 + (-1.0/(4000**2)*(x-4000)**2+1.0)


#calucate the probability for each view: (logit-x0)*beta
#shopping_df = simulate_shopping_information(num_customer,num_product,customer_information_df)
#product_information_df = simulate_product_information(num_product,a,b,p_0,p_1,p_2)
def each_customer_buying(buying_id,beta,x0,shopping_df,product_information_df):
    buying_prob = []
    for i in range(int(shopping_df.loc[buying_id][2])):        
        gender = shopping_df.loc[buying_id][1]
        x = shopping_df.loc[buying_id][3+int(i)]
        category = product_information_df.loc[int(x)]['category']    
        price = product_information_df.loc[int(x)]['prices']        
        if gender == 0 and category == 0:
            logit = lr_m0(price)
        elif gender == 0 and category == 1:
            logit = lr_m1(price)
        elif gender == 0 and category == 2:
            logit = lr_m2(price)
        elif gender == 1 and category == 0:
            logit = lr_f0(price)
        elif gender == 1 and category == 1:
            logit = lr_f1(price)
        elif gender == 1 and category == 2:
            logit = lr_f2(price)             
        P_buy = 1/(1+math.exp(-(logit-x0)*beta))
        buying_prob.append(P_buy)        
    return buying_prob  

# determine the product is brought or not! P>=0.5 brought.
def each_customer_buy_0_1(buying_id,beta,x0,shopping_df,product_information_df):
    buying_or_not = []    
    for i in range(int(shopping_df.loc[buying_id][2])):
        gender = shopping_df.loc[buying_id][1]
        x = shopping_df.loc[buying_id][3+int(i)]
        category = product_information_df.loc[int(x)]['category']    
        price = product_information_df.loc[int(x)]['prices']
        if gender == 0 and category == 0:
            logit = lr_m0(price)
        elif gender == 0 and category == 1:
            logit = lr_m1(price)
        elif gender == 0 and category == 2:
            logit = lr_m2(price)
        elif gender == 1 and category == 0:
            logit = lr_f0(price)
        elif gender == 1 and category == 1:
            logit = lr_f1(price)
        elif gender == 1 and category == 2:
            logit = lr_f2(price)       
        P_buy = 1/(1+math.exp(-(logit-x0)*beta))        
        if P_buy >= 0.5:
            buying_or_not.append(1)
        else:
            buying_or_not.append(0)        
    return buying_or_not

"""
Part 5:(simulated_buying),(simulater_buying_0_1)
Combine total probality and buy or not buy into a table 
"""
# calulate the total probability for all viewed product
def simulate_total_prob(beta,x0,shopping_df,product_information_df):
    total_buying_prob = []
    for i in range(num_customer):
        x = each_customer_buying(i,beta,x0,shopping_df,product_information_df)
        total_buying_prob.append(x)
    total_prob_df = pd.DataFrame(total_buying_prob)
    return total_prob_df

# determine which viewed product is brought 
# and calculate the number of product brought in each session 
def simulate_total_buying01(beta,x0,shopping_df,product_information_df):
    total_buying_0_1 = []
    for i in range(num_customer):
        x = each_customer_buy_0_1(i,beta,x0,shopping_df,product_information_df)
        total_buying_0_1.append(x)
    total_0_1_df = pd.DataFrame(total_buying_0_1)
    return total_0_1_df

beta=0.1
x0 = 8
total_prob_df = simulate_total_prob(beta,x0,shopping_df,product_information_df)
total_0_1_df = simulate_total_buying01(beta,x0,shopping_df,product_information_df)
total_0_1_df = total_0_1_df.fillna("")

# the buying probability for each customer at each session 
simulated_buying = pd.concat([customer_information_df,total_prob_df], axis=1)
# buy or not 
simulated_buying_0_1 = pd.concat([customer_information_df,total_0_1_df], axis=1)

print("The bought product information table: ")
simulated_buying_0_1.loc[0:5]

"""
Part 6:(whole_data)
Data transformation:
for each row: transform each session to the row 
"""
def simulate_whole_data(num_customer,simulated_buying_0_1,shopping_df):
    whole_data = pd.DataFrame()
    for item in list(range(0,num_customer)):
        id = item
        customer_list = []
        gender_list = []
        session_length_list = []
    
        #change the gender 0-M; 1-F
        if simulated_buying_0_1['Gender'][id] == 0:
            for i in range(simulated_buying_0_1['session_length'][id]):
                gender_list.append("M")
        else:
            for i in range(simulated_buying_0_1['session_length'][id]):
                gender_list.append("F")
    
        #get the product id 
        for i in range(simulated_buying_0_1['session_length'][id]):
            customer_list.append(int(shopping_df.loc[id][0]))
       
        #get the orginal index
        for i in range(simulated_buying_0_1['session_length'][id]):
            session_length_list.append(id)
        
        buy01 = simulated_buying_0_1.loc[id][3:int(3+simulated_buying_0_1['session_length'][id])]
        product_id_i = shopping_df.loc[id][3:int(3+simulated_buying_0_1['session_length'][id])]
        df_i = pd.DataFrame(np.column_stack((customer_list,gender_list,product_id_i,buy01,session_length_list)))
        whole_data = whole_data.append(df_i)
    
    whole_data.columns = ["Customer","Gender","Product","Brought","index"]
    whole_data = pd.DataFrame(whole_data)
    whole_data['Customer'] = whole_data['Customer'].astype(int)
    whole_data['Gender'] = whole_data['Gender'].astype(str)
    whole_data['Product'] = whole_data['Product'].astype(float).astype(int)
    whole_data['Brought'] = whole_data['Brought'].astype(float).astype(int)
    whole_data['index'] = whole_data['index'].astype(int)
    
    whole_data.reset_index(drop=True, inplace=True)
    whole_data["Session"] = whole_data.groupby(["Customer"])["index"].rank(method = "dense",ascending=True)
    whole_data["Session"] = whole_data["Session"].astype(int)
    whole_data.drop('index', axis=1, inplace=True)
    
    return whole_data

whole_data = simulate_whole_data(num_customer,simulated_buying_0_1,shopping_df)
whole_data.dtypes

whole_data.loc[0:5]

"""
Part 7:(whole_data_format)
Data transformation:
for each row: transform each session to the row 
"""
#calculate the total brought and view 
def simulate_data_format(whole_data):
    whole_data_format = whole_data[["Customer","Gender","Product","Brought","Session"]].    groupby(["Customer","Gender","Session","Product"]).agg(['sum', 'count'])
    
    return whole_data_format



whole_data_format = simulate_data_format(whole_data)

whole_data.sort_values(by=['Session'], ascending=False)[0:5]

pd.value_counts(shopping_df['customer_id'])[0:5]

#whole_data_format

whole_data_format









