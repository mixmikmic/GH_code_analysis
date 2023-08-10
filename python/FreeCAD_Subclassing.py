import Part
class MyFeature(Part.BSplineSurface):
    pass

import FreeCAD as App
class MyVector(App.Rotation):
    pass

import eigen
class MyEigenVector(eigen.vector3):
    def size(self):
        return 4

a = MyEigenVector([1,2,3])
b = eigen.vector3([1,2,3])

a.size()

b.size()

len(a)

len(b)



