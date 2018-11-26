# -*-coding:utf-8-*-

import types

class Fly(object):
    def fly(self):
        print('flying')

class Walk(object):
    def walk(self):
        print('Walking')

class Swim(object):
    def swim(self):
        print('Swimming')


# class Bird(Fly):
#     def canFly(self):
#         attr=self.__getattribute__('fly')
#         if attr != None:
#             print(type(attr))
#             return True
#         return False

class Animal:
    def canFly(self):
        return self.hasFunc('fly')

    def canWalk(self):
        return self.hasFunc('walk')

    def canSwim(self):
        return self.hasFunc('swim')

    def play(self):
        if self.canFly():
            self.fly()
        if self.canWalk():
            self.walk()
        if self.canSwim():
            self.swim()

    def hasFunc(self, name):
        try:
            attr = self.__getattribute__(name)
            if attr != None and type(attr) == types.MethodType:
                return True
            return False
        except:
            return False


class Bird(Animal):
    pass

class Frog(Animal):
    pass


def mixin(pyClass, pyMixinClass, key=1):
    if key:
        pyClass.__bases__ = (pyMixinClass,) + pyClass.__bases__
    elif pyMixinClass not in pyClass.__bases__:
        pyClass.__bases__ += (pyMixinClass,)
    else:
        pass


mixin(Bird,Fly)
bird=Bird()
print('bird playing:')
bird.play()

mixin(Frog,Walk)
mixin(Frog,Swim)
frog=Frog()
print('frog playing:')
frog.play()

