import pandas, numpy
S = numpy.matrix(pandas.read_excel('Tut8eq.xlsx'))
S1 = numpy.vstack([S,[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0]]) # Adding mu and zero PFL spec.



