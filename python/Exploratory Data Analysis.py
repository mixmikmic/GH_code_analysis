import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

merge_file = pd.read_csv("FY2010_merge_all.csv", encoding = "ISO-8859-1") 
merge_file = merge_file.drop("Unnamed: 0", axis=1)
merge_file.head()

### clean data and remove empty values.
merge_file_dropna = merge_file.dropna()
### comapre the number of publications before and after remove na value
total_publications_before = len(merge_file)
total_publications_after = len(merge_file_dropna)
print("Before remove na, there are", total_publications_before , "pblications.")
print("Before remove na, there are", total_publications_after , "pblications.")

merge_file_dropna_total_grant = merge_file_dropna[['GRID', 'TOTAL_COST']].groupby(['GRID']).size()
total_grant = len(merge_file_dropna_total_grant)

merge_file_dropna_total_cost = merge_file_dropna[['GRID', 'TOTAL_COST']].groupby(['GRID']).mean().sort_values(by='TOTAL_COST', ascending=True)
total_cost = sum(merge_file_dropna_total_cost["TOTAL_COST"])

average_cost = total_cost/total_publications_after

print("The total number of new grant in 2010 is", total_grant)
print("The total cost of new grant in 2010 is", total_cost, "US dollars." )
print("The total number of publications of new grant in 2010 is", total_publications_after )
print("The average cost of publications of new grant in 2010 is", average_cost,"US dollars.")

GRID = merge_file_dropna["GRID"]
R01 = [x for x in list(GRID) if x.startswith("R01")]

R01_df = merge_file_dropna[merge_file_dropna["GRID"].isin(R01)]
all_publications = len(merge_file_dropna)
R01_publications = len(R01_df)

print("The number of publications is", all_publications)
print("The number of R01 related publications is", R01_publications)

R01_total_cost = R01_df[['GRID', 'TOTAL_COST']].groupby(['GRID']).mean().sort_values(by='TOTAL_COST', ascending=True)
R01_total_cost.tail()

R01_total_grant = R01_df[['GRID', 'TOTAL_COST']].groupby(['GRID']).size()
R01_total_grant_number = len(R01_total_grant)
R01_tota_cost = sum(R01_total_cost["TOTAL_COST"])
R01_ave_cost = R01_tota_cost/R01_publications

print("The total number of R01 new grant in 2010 is", R01_total_grant_number)
print("The total cost of R01 new grant in 2010 is", R01_tota_cost,"US dollars.")
print("The total number of R01 publications of new grant in 2010 is", R01_publications)
print("The average cost of R01 publications of new grant in 2010 is", R01_ave_cost,"US dollars.")

import numpy 
import pandas as pd

with open(file,"r") as f:
    header= f.readline()
    h1=header.strip().split(",")
    h1.append("GR_Star_year")
    h1.append("GR_End_year")
    h1.append("GR_Total_year")
    h1.append("Pub_year")
    h1.append("time_to_pub")
    new_header=",".join(h1)
    output.write(new_header+"\n")
    
    rest=f.readlines()
    for line in rest:
        newline=line.strip().split(",")
        Str_year=newline[3].strip().split("/")[-1]
        Str_year="20"+Str_year
        End_year=newline[4].strip().split("/")[-1]
        End_year="20"+End_year
        Total_year=int(End_year)-int(Str_year)
        Pub_year=newline[10].strip().split(" ")[0]
        time_to_pub=int(Pub_year)-int(2010)
        
        newline.append(str(Str_year))
        newline.append(str(End_year))
        newline.append(str(Total_year))
        newline.append(str(Pub_year))
        newline.append(str(time_to_pub))
        newline2=",".join(newline) 
        output.write(newline2+"\n")
        print(newline)

f.close()
output.close()

df_add_time = pd.read_csv("FY2010_merge_add_features.csv", encoding = "ISO-8859-1") 
df_add_time_drop = df_add_time.drop("Unnamed: 0", axis=1)
df_add_time_drop.head()

df_add_time_drop_bytime = df_add_time_drop[["CORE_PROJECT_NUM","time_to_pub"]].groupby(["CORE_PROJECT_NUM"]).mean().sort_values(by="time_to_pub",ascending=False)
average_time = df_add_time_drop_bytime["time_to_pub"].mean()
print("The average time of publish in 2010 is ", average_time, "years")

df_by_insitute = merge_file_dropna[['ADMINISTERING_IC', 'TOTAL_COST']].groupby(['ADMINISTERING_IC']).median().sort_values(by='TOTAL_COST', ascending=False)
df_by_insitute.reset_index(level=0, inplace=True)
df_by_insitute.reset_index(level=0, inplace=True)
df_by_insitute.head(6)

## generate scatter plot
df_by_insitute.plot.scatter(x='index', y='TOTAL_COST',color="blue",alpha=0.6)
plt.title('Total cost of publications')
plt.xlabel('NIH insitute')
plt.ylabel('TOTAL_COST')
plt.annotate('HM', xy=(1,16.1e6),xytext=(1,16.1e6),color="blue")
plt.annotate('PS', xy=(0.5,25e5),xytext=(0.5,25e5),color="blue")
plt.annotate('TP', xy=(1.5,17e5),xytext=(1.5,17e5),color="blue")
plt.savefig("scatter_total_cost.pdf")
plt.show()

total_pub_by_insitute = merge_file_dropna[['ADMINISTERING_IC', 'PMID']].groupby(['ADMINISTERING_IC']).count().sort_values(by='PMID', ascending=False)
total_pub_by_insitute.reset_index(level=0, inplace=True)
total_pub_by_insitute.reset_index(level=0, inplace=True)
total_pub_by_insitute.head()

total_pub_by_insitute.plot.scatter(x='index', y='PMID',color="red",alpha=0.6)
plt.title('Total number of publications')
plt.xlabel('NIH insitute')
plt.ylabel('paper counts')
plt.annotate('CA', xy=(0.5,3000),xytext=(0.5,3000),color="red")
plt.annotate('HL', xy=(0.5,3000),xytext=(1.5,2000),color="red")
plt.annotate('DK', xy=(1.5,1500),xytext=(1.5,1500),color="red")
plt.savefig("scatter_total_pub.pdf")
plt.show()

publish_speed = df_add_time_drop[["ADMINISTERING_IC","time_to_pub"]].groupby(["ADMINISTERING_IC"]).mean().sort_values(by="time_to_pub",ascending=True)
publish_speed.reset_index(level=0, inplace=True)
publish_speed.reset_index(level=0, inplace=True)
publish_speed.head()

publish_speed.plot.scatter(x='index', y='time_to_pub',color="purple",alpha=0.6)
plt.title('average time to publish')
plt.xlabel('NIH insitute')
plt.ylabel('publication time (years)')
plt.annotate('HG', xy=(1,3),xytext=(0.5,2.9),color="purple")
plt.annotate('EB', xy=(0.5,3.05),xytext=(0.5,3.05),color="purple")
plt.annotate('AI', xy=(1.5,3.25),xytext=(1.5,3.25),color="purple")
plt.savefig("scatter_speed.pdf")
plt.show()

pub = total_pub_by_insitute.copy()
time = publish_speed.copy()
cost = df_by_insitute.copy()

merge1 = pd.merge(pub,time, on="ADMINISTERING_IC")
merge2 = pd.merge(merge1,cost, on="ADMINISTERING_IC")
merge2.head()

merge2["average"] = merge2["TOTAL_COST"]/merge2["PMID"]
merge3 = merge2.drop("index_x",axis=1)
merge3.to_csv("Insitute_merge.csv")
merge3

import matplotlib.pyplot as plt
import pandas as pd
# '#d95f02' is orange; #1b9e77 is green; #7570b3 is blue; #e7298a is pink; #66a61e is light green.
test = merge2
color = ['#d95f02', '#1b9e77', '#1b9e77', '#7570b3', '#1b9e77', '#e7298a', '#d95f02', '#d95f02', '#d95f02', '#d95f02', '#d95f02', '#d95f02', '#d95f02', '#1b9e77', '#d95f02', '#d95f02', '#e7298a', '#1b9e77', '#d95f02', '#66a61e', '#66a61e', '#66a61e', '#66a61e', '#66a61e', '#66a61e', '#66a61e', '#66a61e','#66a61e', '#66a61e', '#66a61e','#66a61e', '#66a61e', '#66a61e']
fig, ax1 = plt.subplots(figsize=(12,9))
ax1.scatter(x = test['TOTAL_COST'],
            y = test['average'],
            s = test['PMID'],
            alpha = 0.3,
            c = color)

ax1.set_ylim([-1000,4000])
ax1.set_xlim([250000, 500000])

plt.annotate('RR', xy=(460000,600),xytext=(460000,600),color="#7570b3")
plt.annotate('GM', xy=(460000,600),xytext=(290000,500),color="#e7298a")
plt.annotate('NR', xy=(460000,600),xytext=(400000,3800),color="#66a61e")
plt.annotate('HG', xy=(460000,600),xytext=(380000,1800),color="#1b9e77")
plt.annotate('CA', xy=(460000,600),xytext=(340000,-300),color="#d95f02")

plt.title('Total cost vs average cost', fontsize=12)
plt.xlabel('Total cost', fontsize=12)
plt.ylabel('Average cost', fontsize=12)
plt.grid(True,linestyle='--', linewidth=0.5)
plt.savefig("scatter_merge.png")



