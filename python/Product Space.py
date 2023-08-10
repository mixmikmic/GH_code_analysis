import sys
sys.path.append("./modules")
import d3plus2 as d3plus
import pandas as pd

# Read in the data
df = pd.read_csv("./sourcedata/hs_test.csv")

# Take a peek
df.head()

# Zero-fill the hs4 column
df.hs4 = df.hs4.astype(str).str.zfill(4)

# Read in colors
colors = pd.read_csv("./sourcedata/hs4_hex_colors_intl_atlas.csv", dtype={"hs4": str})

# Take a peek
colors.head()

# Grab only the columns we need
colors = colors[["hs4", "color"]]

df = df.merge(
    colors, # Merge our data with colors,
    left_on="hs4", # Using the hs4 column of 'df' (i.e. 'left')
    right_on="hs4", # And using the hs4 column of 'colors' (i.e. 'right')
    how="left", # Keeping all columns on the left, and having nulls on the right for missing values
)

ps = d3plus.ProductSpace(
    id="hs4",
    presence="M",
    color="color",
    name="name_en",
    size='exports'
)
ps.draw(df)

# Save
open("./my_product_space.html", "w+").write(ps.dump_html(df))



