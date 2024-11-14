# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 01:24:15 2023

@author: Leonid
"""
class list_variable:
    def __init__(self, name='x') -> None:
        self.name=name
    def eval(self, param=None):
        try:
            return param[self.name]
        except:
            print("Ошибка вычисления узла типа list_variable: ", self.name)
            raise RuntimeError("Ошибка вычисления узла типа list_variable: ".format(self.name))
        return False

class list_sum:
    def __init__(self, num_childs=2 ) -> None:
        self.name='sum'
        self.num_childs=num_childs
    def eval(self, childs=None, param=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return sum(childs)
        except:
            print("Ошибка вычисления узла типа list_sum: ", self.name)
            raise RuntimeError("Ошибка вычисления узла типа list_sum: name={0}, childs={1}".format(self.name,childs ))
        return False