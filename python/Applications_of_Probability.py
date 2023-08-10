from sie import *

data=randint(2,size=10)
print data

data=randint(2,size=30)
print data

data=randint(2,size=(2000,10))
data

sum(data)  # add up all of the 1's

sum(data,axis=0)  # sum up all of the columns

sum(data,axis=1)  # sum up all of the rows

N=sum(data,axis=1)  # number of heads in each of many flips
hist(N,countbins(10))
xlabel('Number of Heads')
ylabel('Number of Flips')

h=array([0,1,2,3,4,5,6,7,8,9,10])

# or...

h=arange(0,11)

p=nchoosek(10,h)* 0.5**h * 0.5**(10-h)

hist(N,countbins(10),normed=True)
plot(h,p,'--o')
xlabel('Number of Heads, $h$')
ylabel('$p(h|N=10)$')

