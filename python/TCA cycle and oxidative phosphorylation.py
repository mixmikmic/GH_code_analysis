import pandas, numpy
pandas.read_excel('TCA_example.xlsx')

M = numpy.matrix(pandas.read_excel('TCA_example.xlsx'))
mu,theta=0.25,0.1
C = numpy.matrix([[0,0,0,0,0,0,theta,mu,0]]).T
r = numpy.linalg.solve(M,C)
r   

rO2=r[7,0].tolist()  #oxygen rate
S = numpy.matrix([[-1,0,0,1,1,0],
                  [-2,-3,0,1.8,0,2],
                  [-1,0,-2,0.5,2,1],
                  [0,-1,0,0.2,0,0],
                  [0,0,0,1,0,0],
                  [0,0,1,0,0,0]])
        
C2 = numpy.matrix([[0,0,0,0,mu,rO2]]).T
r2 = numpy.linalg.solve(S,C2)
r2      

alpha=0.1
beta=0.1   # Can be calculated - see section 4.1
gamma=2.5
mu,theta = 0.25, 0.1

S = numpy.matrix([[-1,(1+alpha),1,0],
[0,beta,2,-2],
[0,-gamma, 2/3,3],                  
[0,1,0,0]])
        
C = numpy.matrix([[0,0,theta,mu]]).T
Y = numpy.linalg.solve(S,C)
Y      



