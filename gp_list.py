'''
28.01.2023 
В данном файле будет реализован типовой лист для дерева GP
В листе только реализуется вычисление текущего узла. Механика вычисления 
Всего дерева реализуется в другом файле

структура терминального узла:
    self.name = имя узла
    self.value = значение для константного узла
    self.num_childs=0 т.к. у терминального узла нет потомков
    eval - функция возвращает значение терминального узла. Если константа, то 
           само значение, если переменная, то подставляем значение переменной из param,
           где params - словарь, который содержит ключ=имя переменной, 
           значение=значение переменной. 
    copy - функция создает полную копию узла
    get_name - функция используется при распечатке (вывода в строку) дерева

структура функционального узла:
    self.name - имя узла
    self.num_childs - количество потомков характерное для этой функции
    eval - функция вычисляет значение функционального узла. в зависимости от 
           типа функции из param подставляем параметры params, если это требуется и 
           childs - подставляем значение дочерних узлов
    


'''



import numpy as np

class list_const:
    def __init__(self, value=0) -> None:
        self.name='const'
        self.value=value
        self.num_childs=0
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        return self.value
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_const(self.value)
        rez.name=self.name
        rez.value=self.value
        rez.num_childs=self.num_childs
        return rez
    def get_name(self):
        return self.name+'_'+str(self.value)
    
#==============================================================================
class list_variable:
    def __init__(self, name='x') -> None:
        self.name=name
        self.num_childs=0
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return params[self.name]
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_variable: name={0}, param={1}, ".format(self.name, params))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_variable()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name
        
#==============================================================================
class list_sum:
    def __init__(self, num_childs=2 ) -> None:
        self.name='sum'
        self.num_childs=num_childs
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return np.sum(childs, axis=0)
        except:
            
            raise RuntimeError(f"Ошибка вычисления узла типа list_sum: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_sum()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs)
#==============================================================================
class list_prod:
    def __init__(self, num_childs=2 ) -> None:
        self.name='prod'
        self.num_childs=num_childs
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return np.prod(childs, axis=0)
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_prod: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_prod()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    
    def get_name(self):
        return self.name+str(self.num_childs)    
#==============================================================================
class list_sin:
    def __init__(self) -> None:
        self.name='sin'
        self.num_childs=1
    def eval(self, childs=None, params=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            return np.sin(childs[0])
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_sin: name={0}, childs={1}".format(self.name,childs ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_sin()
        rez.name=self.name
        rez.num_childs=self.num_childs
        return rez
    def get_name(self):
        return self.name





































