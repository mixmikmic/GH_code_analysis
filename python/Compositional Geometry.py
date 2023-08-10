# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")
import compositional

D = compositional.Diagram()

D.show()

D.loop_animation()



