get_ipython().magic('run jacobi.ipynb')
get_ipython().magic('run gauss_seidel.ipynb')
get_ipython().magic('run sor.ipynb')
get_ipython().magic('matplotlib inline')

n = 100
A = np.identity(n)*3

for i in range(0,n-1):
    A[i,i+1] = -1
    A[i+1,i] = -1
#print("A = ")
#print(A)

b = np.ones((n,1))
b[0] = 2
b[-1] = 2
#print("b = ")
#print(b)

tol = 10**-6
w = 1.13
[x1, step1, err1] = jacobi(A,b,tol)
[x2, step2, err2] = gauss_seidel(A,b,tol)
[x3, step3, err3] = sor(A,b,tol,w)

plt.figure(figsize=(10,5))
plt.plot(err1,'b')
plt.plot(err2,'r')
plt.plot(err3,'g')
plt.title("Comparing Iterative Methods")
plt.xticks(np.arange(0,len(err1)))
plt.xlabel("step")
plt.ylabel("error")
plt.legend(['Jacobi','Gauss','SOR'])

print("Number of steps in Jacobi: ")
print(step1)
print("Number of steps in Gauss-Seidel: ")
print(step2)
print("Number of steps in SOR: ")
print(step3)



