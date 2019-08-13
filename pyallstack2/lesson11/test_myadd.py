
from .myadd import myadd
from pyallstack2.lesson8.apitest import ApiDemoTest
import unittest

class TestMyAdd(unittest.TestCase):
    def setUp(self):
        print('start test:')

    def tearDown(self):
        print('end test')

    def test_01(self):
        expect_result = 4
        actural_result =  myadd(1,3)
        self.assertEqual(expect_result,actural_result,msg='1加3结果不正确')

    @unittest.skipUnless(2 > 1, 'unless 2>1 skip test_04')
    def test_02(self):
        expect_result = 2
        actural_result =  myadd(-1,3)
        self.assertEqual(expect_result,actural_result,msg='-1加3结果不正确')

    @unittest.skip('skip test_03')
    def test_03(self):
        expect_result = 4
        actural_result =  myadd(1.5,3)
        self.assertEqual(expect_result,actural_result,msg='1加3结果不正确')

    @unittest.skipIf(2>1, '2>1 skip test_04')
    def test_04(self):
        expect_result = 11
        actural_result =  myadd(8,3)
        self.assertEqual(expect_result,actural_result,msg='8加3结果不正确')

    @unittest.expectedFailure
    def test_05(self):
        actural_result =  myadd("abc",3)
        self.assertIsNone(actural_result,msg='abc加3结果不正确')


if __name__ == '__main__':
    # unittest.main()
    testSuite=unittest.TestSuite()
    testSuite.addTest(TestMyAdd())
    # suite = unittest.defaultTestLoader.discover('../lesson8/', pattern="apit*.py")
    testSuite.addTest(ApiDemoTest())
    runner = unittest.TextTestRunner()
    runner.run(testSuite)