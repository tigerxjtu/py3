import unittest
from pytest.lesson1 import op

class op_testcase(unittest.TestCase):

    def setUp(self):
        self._op=op()

    def test_op(self):
        self.assertEqual(2,self._op.add(1,1))
        self.assertEqual(2, self._op.sub(3, 1))
        self.assertEqual(6, self._op.mul(2, 3))
        self.assertEqual(2, self._op.div(6, 3))

if __name__ == "__main__":
    unittest.main()