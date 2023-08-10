from sympy import symbols, cos, sin, pi, simplify, trigsimp, expand_trig, pprint, sqrt, atan2
from sympy.matrices import Matrix

def rotx(q):

  sq, cq = sin(q), cos(q)

  r = Matrix([
    [1., 0., 0.],
    [0., cq,-sq],
    [0., sq, cq]
  ])
    
  return r


def roty(q):

  sq, cq = sin(q), cos(q)

  r = Matrix([
    [ cq, 0., sq],
    [ 0., 1., 0.],
    [-sq, 0., cq]
  ])
    
  return r


def rotz(q):

  sq, cq = sin(q), cos(q)

  r = Matrix([
    [cq,-sq, 0.],
    [sq, cq, 0.],
    [0., 0., 1.]
  ])
    
  return r

q1, q2, q3, q4, q5, q6= symbols('q1:7')

R03 = Matrix([
[sin(q2 + q3)*cos(q1), cos(q1)*cos(q2 + q3), -sin(q1)],
[sin(q1)*sin(q2 + q3), sin(q1)*cos(q2 + q3),  cos(q1)],
[        cos(q2 + q3),        -sin(q2 + q3),        0]])

R03T = R03.T
pprint(R03T)

R36 = Matrix([[-sin(q4)*sin(q6) + cos(q4)*cos(q5)*cos(q6), -sin(q4)*cos(q6) - sin(q6)*cos(q4)*cos(q5), -sin(q5)*cos(q4)],
  [                           sin(q5)*cos(q6),                           -sin(q5)*sin(q6),          cos(q5)],
  [-sin(q4)*cos(q5)*cos(q6) - sin(q6)*cos(q4),  sin(q4)*sin(q6)*cos(q5) - cos(q4)*cos(q6),  sin(q4)*sin(q5)]])

alpha, beta, gamma = symbols('alpha beta gamma', real = True)
R0u = rotz(alpha) * roty(beta) * rotx(gamma)
pprint(R0u)

RugT = (rotz(pi) * roty(-pi/2)).T
pprint(RugT)

R36 = R03T * R0u * RugT
pprint(R36)

q1, q2, q3, q4, q5, q6= symbols('q1:7')

R03 = Matrix([
[sin(q2 + q3)*cos(q1), cos(q1)*cos(q2 + q3), -sin(q1)],
[sin(q1)*sin(q2 + q3), sin(q1)*cos(q2 + q3),  cos(q1)],
[        cos(q2 + q3),        -sin(q2 + q3),        0]])

R03T = R03.T

alpha, beta, gamma = symbols('alpha beta gamma')
R0u = rotz(alpha) * roty(beta) * rotx(gamma)

RugT = (rotz(pi) * roty(-pi/2)).T
R36 = R03T * R0u * RugT


roll, pitch, yaw = 0.366, -0.078, 2.561

variables = {
  q1: 1.01249, 
  q2: -0.2758, 
  q3: -0.11568,
  alpha: yaw,
  beta: pitch, 
  gamma: roll
}

R0g = R0u * Rug
R36eval = R36.evalf(subs = variables)

pprint(R36eval)

print(get_spherical_ik(R36eval))

print(simplify(R03I) == R03.T)
print(Rug == Rug.T)

R03 = Matrix([
[sin(q2 + q3)*cos(q1), cos(q1)*cos(q2 + q3), -sin(q1)],
[sin(q1)*sin(q2 + q3), sin(q1)*cos(q2 + q3),  cos(q1)],
[        cos(q2 + q3),        -sin(q2 + q3),        0]])

roll, pitch, yaw = 0.366, -0.078, 2.561
R0u = rotz(yaw)* roty(pitch) * rotx(roll)
Rug = (rotz(pi) * roty(-pi/2)).T

R0u.evalf(subs = {alpha: yaw, beta: pitch, gamma: roll})
R36eval2 = R36.evalf(subs = {q1: 1.01249, q2: -0.2758, q3: -0.11568})
pprint(R36eval2)



def ik(R):
  r12, r13 = R[0,1], R[0,2]
  r21, r22, r23 = R[1,0], R[1,1], R[1,2] 
  r32, r33 = R[2,1], R[2,2]
  q5 = atan2(sqrt(r13**2 + r33**2), r23)
  q4 = atan2(r33, -r13)
  q6 = atan2(-r22, r21)
  return q4.evalf(), q5.evalf(), q6.evalf()

print(ik(R36eval))

print(R36eval)

R0g = rotz(yaw) * roty(pitch) * rotx(roll) * Rug
print(R0g)

R03T_eval =R03T.evalf(subs = { q1: 1.01249, q2: -0.2758, q3: -0.11568})
pprint(R03T_eval)
pprint(R03T)



