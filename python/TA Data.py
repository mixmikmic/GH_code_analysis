import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn
seaborn.set()

mean_rents = pd.read_csv('..\\data\\Detailed Geomean Rents.csv')

mean_rents.head()

print(list(mean_rents.columns))

date_cols = ['1993-03-01', '1993-06-01', '1993-09-01', '1993-12-01', '1994-03-01', '1994-06-01', '1994-09-01', '1994-12-01', '1995-03-01', '1995-06-01', '1995-09-01', '1995-12-01', '1996-03-01', '1996-06-01', '1996-09-01', '1996-12-01', '1997-03-01', '1997-06-01', '1997-09-01', '1997-12-01', '1998-03-01', '1998-06-01', '1998-09-01', '1998-12-01', '1999-03-01', '1999-06-01', '1999-09-01', '1999-12-01', '2000-03-01', '2000-06-01', '2000-09-01', '2000-12-01', '2001-03-01', '2001-06-01', '2001-09-01', '2001-12-01', '2002-03-01', '2002-06-01', '2002-09-01', '2002-12-01', '2003-03-01', '2003-06-01', '2003-09-01', '2003-12-01', '2004-03-01', '2004-06-01', '2004-09-01', '2004-12-01', '2005-03-01', '2005-06-01', '2005-09-01', '2005-12-01', '2006-03-01', '2006-06-01', '2006-09-01', '2006-12-01', '2007-03-01', '2007-06-01', '2007-09-01', '2007-12-01', '2008-03-01', '2008-06-01', '2008-09-01', '2008-12-01', '2009-03-01', '2009-06-01', '2009-09-01', '2009-12-01', '2010-03-01', '2010-06-01', '2010-09-01', '2010-12-01', '2011-03-01', '2011-06-01', '2011-09-01', '2011-12-01', '2012-03-01', '2012-06-01', '2012-09-01', '2012-12-01', '2013-03-01', '2013-06-01', '2013-09-01', '2013-12-01', '2014-03-01', '2014-06-01', '2014-09-01', '2014-12-01', '2015-03-01', '2015-06-01', '2015-09-01', '2015-12-01', '2016-03-01', '2016-06-01', '2016-09-01', '2016-12-01', '2017-03-01', '2017-06-01']

mean_rents.fillna(0)    .groupby("SAU")[date_cols]    .sum()    .transpose()    .rolling(window=15).mean()    .transpose()

mean_rents.columns[3:]

mean_rents.ix[mean_rents.SAU == '574702', :]

