import unittest
from internal_displacement.tests.test_model import TestModel
suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
unittest.TextTestRunner(verbosity=3).run(suite)



