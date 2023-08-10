import os, sys
sys.path.append(os.path.abspath('../../main/python'))
from thalesians.tsa.conditions import precondition, postcondition

class Subtractor(object):
    @precondition(lambda self, arg1, arg2: arg1 >= 0, 'arg1 must be greater than or equal to 0')
    @precondition(lambda self, arg1, arg2: arg2 >= 0, 'arg2 must be greater than or equal to 0')
    @postcondition(lambda result: result >= 0, 'result must be greater than or equal to 0')
    def subtract(self, arg1, arg2):
        return arg1 - arg2

subtractor = Subtractor()
subtractor.subtract(300, 200)

subtractor.subtract(-300, 200)

MIN_PRECONDITION_LEVEL = 5
MIN_POSTCONDITION_LEVEL = 7

