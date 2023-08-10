import pandas as pd

bond = pd.read_csv("jamesbond.csv")
bond

bond.set_index("Film", inplace= True)

bond.loc["Goldfinger"]

bond = pd.read_csv("jamesbond.csv")

bond.loc[15]
bond.iloc[15]

bond.set_index("Film", inplace= True)
bond.sort_index(inplace = True)

bond.iloc[14]

bond.loc["Octopussy"]

bond.ix["Octopussy"]

bond.ix[14]

bond.loc["Octopussy",["Actor","Year"]]

bond.iloc[14,[1,2,4]]

bond.iloc[14,[1]] = "Sir Roger Moore"

bond["Actor"].unique()

mask = bond["Actor"] == "Sean Connery"

bond[mask]["Actor"] = "Sir Sean Connery"

df2 = bond["Actor"]
df2[mask]["Actor"] = "Sir Sean Connery"

df2

bond = pd.read_csv("jamesbond.csv")
bond.rename(columns={"Year":"Release Date"})

bond.columns

bond.drop("A View To Kill")

bond.drop("Box Office", axis = 1)

bond = pd.read_csv("jamesbond.csv")

bond.sample()

bond.sample(5)

bond.sample(frac= .25)

bond.nlargest(n = 4 , columns= "Box Office")

bond = pd.read_csv("jamesbond.csv")


mask = bond["Actor"] == "Sean Connery"
bond[mask]

bond.where()

bond.columns = [column_name.replace(" ","_") for column_name in bond.columns]

bond.head()

bond.query("Box_Office > 500")

def convert_to_millions(number):
    return str(number) + " Millions!"

bond["Box Office"] = bond["Box Office"].apply(convert_to_millions)
bond["Box Office"]

directors = bond["Director"]

directors = bond["Director"].copy()

directors

directors["Sam Mendes"] = "Sit Sam Mendes"

directors

bond



