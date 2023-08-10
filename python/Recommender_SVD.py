get_ipython().magic('pylab inline')

A = array([[2,-2,1,0,-1],
           [2,-1,0,-2,0],
           [0,2,-1,0,0],
           [-1,2,0,1,2]])
print A

(U,S,Vstar) = svd(A)

A1 = dot(U[:,0:1], S[0]*Vstar[0:1,:])
print A1

U[:,0:1]

Vstar[0:1,:].T

figure(figsize=(12,5))
subplot(1,2,1)
pcolor(flipud(array(A)), cmap=cm.RdYlBu, edgecolors='k')
xlabel('People')
ylabel('Movies')
xticks([])
yticks([])
title('Original data')
colorbar()

subplot(1,2,2)
pcolor(flipud(array(A1)), cmap=cm.RdYlBu, edgecolors='k')
xlabel('People')
ylabel('Movies')
xticks([])
yticks([])
title('Rank 1 approximation')
colorbar()

print "Singular values are: ",S



