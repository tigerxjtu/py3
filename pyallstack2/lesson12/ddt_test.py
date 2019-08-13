import unittest
import ddt


@ddt.ddt
class MyTestCase(unittest.TestCase):
    # 下面的1,2,3代表我们传入的参数,每次传入一个值
    @ddt.data(1, 2, 3)
    # 定义一个value用于接收我们传入的参数
    def test_something(self, value):
        self.assertEqual(value, 2)

    @ddt.data([3, 2], [4, 3], [5, 3])
    @ddt.unpack
    def test_list_extracted_into_arguments(self, first_value, second_value):
        self.assertTrue(first_value > second_value)

    @ddt.data({'first': 1, 'second': 3, 'third': 2},
              {'first': 4, 'second': 6, 'third': 5})
    @ddt.unpack
    def test_dicts_extracted_into_kwargs(self, first, second, third):
        self.assertTrue(first < third < second)

if __name__ == '__main__':
    unittest.main()
