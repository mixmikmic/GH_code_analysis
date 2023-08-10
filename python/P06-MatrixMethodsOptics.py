get_ipython().magic('pylab inline')

ng=1.5  # index of glass
na=1.0  # index of air
h=0.01  # start with ray 1.0 cm above axis
R=0.15  # radius of curvature of lens

ray1 = array([[h],[0.0]])

ray1

T1 = array([[1.0, 3.0],
            [0.0, 1.0]
            ])
T1

ray2 = T1.dot(ray1)  # matrix multiplication is handled by the "dot" method of an array
ray2

R1 = array([[1.0, 0.0],               # entering the curved surface
            [(na-ng)/(ng*R), na/ng]])
R1

ray3 = R1.dot(ray2)
ray3

R2 = array([[1.0,0.0],              # exiting the planer surface
            [0.0, ng/na]])
R2

ray4=R2.dot(ray3)
ray4

fl=-ray4[0,0]/ray4[1,0]         # calculate the focal length from the height and angle of the ray.
fl

na*R/(ng-na)                    # compare to the "lens makers" equation result.

M = R2.dot(R1.dot(T1))   # system matrix
M

M.dot(ray1)             # system acting on ray1



