get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[7,8,9]},index=["one", "two", "three"])

get_ipython().run_line_magic('R', '-i df')

get_ipython().run_cell_magic('R', '', 'df <- as.data.frame(df)\nlibrary(ggplot2)\np <- ggplot(df, aes(A, B))\np <- p + \n    geom_point()\nprint(p)')

get_ipython().run_cell_magic('R', "-w 480 -h 300 -u px # instead of px, you can also choose 'in', 'cm', or 'mm'", 'df <- as.data.frame(df)\nlibrary(ggplot2)\np <- ggplot(df, aes(A, B))\np <- p + \n    geom_point()\nprint(p)')

get_ipython().run_cell_magic('R', '', 'library(shiny)\nrunExample("01_hello")')

