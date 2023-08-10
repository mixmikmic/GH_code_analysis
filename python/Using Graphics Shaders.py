# Creating the Uniview data file
nring = 36
ntube = 25
dTheta = 360./nring
dl = 1.0
writefile=open('GWcatepillar/tube.raw',"w")
for i in range(nring):
    for j in range (-ntube,ntube+1):
        writefile.write("{} {} {} {} {} {} {} {} {}\n".format(i*dTheta,j*dl,i,j,nring,ntube,(i+1)*dTheta,min(j+1,ntube-1)*dl,"1"))
writefile.flush()
writefile.close()



