import math
xyz_volume = math.sqrt(2)**3
ivm_volume = 3
print("XYZ units:", xyz_volume)
print("IVM units:", ivm_volume)
print("Conversion constant:", ivm_volume/xyz_volume)

from math import sqrt, hypot

class Tetrahedron:
    """
    Takes six edges of tetrahedron with faces
    (a,b,d)(b,c,e)(c,a,f)(d,e,f) -- returns volume
    in ivm and xyz units
    """

    def __init__(self, a,b,c,d,e,f):
        self.a, self.a2 = a, a**2
        self.b, self.b2 = b, b**2
        self.c, self.c2 = c, c**2
        self.d, self.d2 = d, d**2
        self.e, self.e2 = e, e**2
        self.f, self.f2 = f, f**2

    def ivm_volume(self):
        ivmvol = ((self._addopen() - self._addclosed() - self._addopposite())/2) ** 0.5
        return ivmvol

    def xyz_volume(self):
        xyzvol = sqrt(8/9) * self.ivm_volume()
        return xyzvol

    def _addopen(self):
        a2,b2,c2,d2,e2,f2 = self.a2, self.b2, self.c2, self.d2, self.e2, self.f2
        sumval = f2*a2*b2
        sumval +=  d2 * a2 * c2
        sumval +=  a2 * b2 * e2
        sumval +=  c2 * b2 * d2
        sumval +=  e2 * c2 * a2
        sumval +=  f2 * c2 * b2
        sumval +=  e2 * d2 * a2
        sumval +=  b2 * d2 * f2
        sumval +=  b2 * e2 * f2
        sumval +=  d2 * e2 * c2
        sumval +=  a2 * f2 * e2
        sumval +=  d2 * f2 * c2
        return sumval

    def _addclosed(self):
        a2,b2,c2,d2,e2,f2 = self.a2, self.b2, self.c2, self.d2, self.e2, self.f2
        sumval =   a2 * b2 * d2
        sumval +=  d2 * e2 * f2
        sumval +=  b2 * c2 * e2
        sumval +=  a2 * c2 * f2
        return sumval

    def _addopposite(self):
        a2,b2,c2,d2,e2,f2 = self.a2, self.b2, self.c2, self.d2, self.e2, self.f2
        sumval =  a2 * e2 * (a2 + e2)
        sumval += b2 * f2 * (b2 + f2)
        sumval += c2 * d2 * (c2 + d2)
        return sumval


PHI = sqrt(5)/2 + 0.5

R =0.5
D =1.0

tet = Tetrahedron(D, D, D, D, D, D)
print(tet.ivm_volume())

import unittest
class Test_Tetrahedron(unittest.TestCase):

    def test_unit_volume(self):
        tet = Tetrahedron(D, D, D, D, D, D)
        print(tet.ivm_volume())
        self.assertAlmostEqual(tet.ivm_volume(), 1.0, places=5)

    def test_unit_volume(self):
        tet = Tetrahedron(R, R, R, R, R, R)
        self.assertAlmostEqual(tet.xyz_volume(), 0.11785, places=5)

    def test_phi_edge_tetra(self):
        tet = Tetrahedron(D, D, D, D, D, PHI)
        self.assertAlmostEqual(tet.ivm_volume(), 0.70711, places=5)

    def test_right_tetra(self):
        e = hypot(sqrt(3)/2, sqrt(3)/2)  # right tetrahedron
        tet = Tetrahedron(D, D, D, D, D, e)
        self.assertAlmostEqual(tet.xyz_volume(), 1.0, places=5)

a = Test_Tetrahedron()

R =0.5
D =1.0

suite = unittest.TestLoader().loadTestsFromModule(a)
unittest.TextTestRunner().run(suite)

a = 2
b = 4
c = 5
d = 3.4641016151377544
e = 4.58257569495584
f = 4.358898943540673
tetra = Tetrahedron(a,b,c,d,e,f)
print("IVM volume of tetra:", tetra.ivm_volume())

b = sqrt(2)/4
a = c = sqrt(3/8)
d = e = 0.5
f = sqrt(2)/2
mite = Tetrahedron(a, b, c, d, e, f)
print("IVM volume of Mite:", mite.ivm_volume())
print("XYZ volume of Mite:", mite.xyz_volume())

regular = Tetrahedron(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
print("MITE volume in XYZ units:", regular.xyz_volume())
print("XYZ volume of 24-Mite Cube:", 24 * regular.xyz_volume())

