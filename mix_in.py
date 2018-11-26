def mixin(pyClass, pyMixinClass, key=0):
    if key:
        pyClass.__bases__ = (pyMixinClass,) + pyClass.__bases__
    elif pyMixinClass not in pyClass.__bases__:
        pyClass.__bases__ += (pyMixinClass,)
    else:
        pass

class test1:
    def test(self):
        print('In the test1 class!')

class testMixin:
    def test(self):
        print('In the testMixin class!')

class test2(test1, testMixin):
    def test(self):
        print('In the test2 class!')

class test0(test1):
    pass

if __name__ == '__main__':
    # print(test0.__mro__)
    # test_0 = test0()
    # test_0.test()  #调用test1的方法

    mixin(test0, testMixin, 0)  #优先继承testMixin类
    test__0 = test0()
    test__0.test()  #由于优先继承了testMixin类，所以调用testMixin类的方法
    print(test0.__mro__)

    # print(test2.__mro__)
    # mixin(test2, testMixin)
    # print(test2.__mro__)