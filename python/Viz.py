# displays your plots without you having to explicitly call show
get_ipython().magic('matplotlib inline')
# have your lab4 code auto reload when you re-run
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import ggplot as gg   # in vagrant VM, do "pip install ggplot" to install ggplot
import pandas as pd

countries = pd.read_csv("cia.csv")
countries = countries.dropna()  # get rid of rows that have missing data
countries.head()

g = gg.ggplot(countries, gg.aes(x="educ", y="gdp")) 
g + gg.geom_point(size=40)   

g = gg.ggplot(countries, gg.aes(x="educ", y="gdp", label="country")) 
g + gg.geom_text()

g = gg.ggplot(countries, gg.aes(x="educ", y="gdp", label="country")) 
g + gg.geom_point() + gg.geom_text(hjust=0.15, vjust=1000, color="red")  # note: text offsets hjust and vjust are based on scale of data

g = gg.ggplot(countries, gg.aes(x="educ", y="gdp", color="net_users", size="roadways")) 
g + gg.geom_point()

g = gg.ggplot(countries, gg.aes(x="educ", y="gdp", color="net_users")) 
g + gg.geom_point() + gg.scale_color_brewer(type="qual") + gg.scale_y_log() 

g = gg.ggplot(countries, gg.aes(x="educ", y="gdp")) 
g + gg.geom_point() + gg.facet_wrap("net_users", scales="fixed")

g = gg.ggplot(countries, gg.aes(x="educ", y="gdp", color="net_users")) 
g + gg.geom_point() + gg.scale_color_brewer(type="qual") + gg.scale_y_log() + gg.ggtitle("Relationship between GDP, Education and Net Usage") + gg.xlab("Education") + gg.ylab("GDP") + gg.theme_bw()

from ggplot import meat

meat.head()   # ha!

g = gg.ggplot(meat, gg.aes()) 
g + gg.geom_line(gg.aes(x="date", y="veal", color="red")) + gg.geom_line(gg.aes(x="date", y="pork", color="blue")) + gg.theme_bw()

meat2 = pd.melt(meat, id_vars=['date'], var_name='meat_type', value_name='quantity')
meat2.head()

meat2.tail()

g = gg.ggplot(meat2, gg.aes(x="date", y="quantity", color="meat_type")) 
g + gg.geom_line() + gg.theme_bw() + gg.scale_color_brewer(type="qual") 



