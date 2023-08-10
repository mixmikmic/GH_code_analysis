get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

get_ipython().run_cell_magic('R', '', 'df = read.csv("/home/pybokeh/Downloads/ch10/SwordDemand2.csv",header=TRUE,stringsAsFactors=FALSE)\nprint(df)')

get_ipython().run_cell_magic('R', '', 'str(df)')

get_ipython().run_cell_magic('R', '', 'library(xts)\ndf$Date = as.yearmon(df$Date, format="%Y-%m") \nstr(df)')

get_ipython().run_cell_magic('R', '', 'dft = ts(df$Qty,frequency=12,start=c(2013,1)) # or dft = zoo(df$qty, order.by=df$date, frequency=12)\nprint(dft)')

get_ipython().run_cell_magic('R', '', 'library(forecast)\nhwm = HoltWinters(dft, gamma=TRUE)\nhwf = forecast.HoltWinters(hwm, h=24)\nsummary(hwf)')

get_ipython().run_cell_magic('R', '', 'plot.forecast(hwf)')

