'''
28.01.2023 
В данном файле будет реализован типовой лист для дерева GP
В листе только реализуется вычисление текущего узла. Механика вычисления 
Всего дерева реализуется в другом файле

структура терминального узла:{
    self.name = имя узла
    self.value = значение для константного узла
    self.num_childs=0 т.к. у терминального узла нет потомков
    eval - функция возвращает значение терминального узла. Если константа, то 
           само значение, если переменная, то подставляем значение переменной из param,
           где params - словарь, который содержит ключ=имя переменной, 
           значение=значение переменной.
           Для ускорения процессса вычисления дерева будем вычислять векторно
           Выход будет записываться в переменную y_pred согласно маске
           Предполагается, что массив y_pred это массив под все выходы выборки. 
           В текущем терминальном узле y_pred вычисляется только для тех значений,
           которые соответствуют маске mask. Т.е. именно этим значениям соответствуют 
           предикаты ведущие к текущему узлу. 
           В этом случае структура params будет следующей
           params={X:выборка входов, y:настоящий выход, 'params':{перечень всех параметров предикатов и терминалов}} 

    copy - функция создает полную копию узла
    get_name - функция используется при распечатке (вывода в строку) дерева

структура функционального узла:
    self.name - имя узла
    self.num_childs - количество потомков характерное для этой функции
    eval - функция вычисляет значение функционального узла. в зависимости от 
           типа функции из param подставляем параметры params, если это требуется и 
           childs - подставляем значение дочерних узлов
    copy - функция создает полную копию узла
    get_name - функция используется при распечатке (вывода в строку) дерева
    


'''



import numpy as np
import  pandas as pd

class list_nom_class:
    def __init__(self, value=0) -> None:
        self.name='nom_class'
        self.value=value
        self.num_childs=0
    def eval(self, childs=None, params=None, mask=None):
        #для общности в каждый узел передаем и потомков и параметры
        # создаем шаблон для выхода
        y_pred=pd.Series(np.nan, index= params['y'].index)

        y_pred.loc[mask]=self.value
        return y_pred
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_nom_class(self.value)
        rez.name=self.name
        rez.value=self.value
        rez.num_childs=self.num_childs
        return rez
    def get_name(self):
        return self.name+'_'+str(self.value)
    
#==============================================================================
class list_regr_const:
    def __init__(self, name='const_0') -> None:
        self.name=name
        self.num_childs=0
    def eval(self, childs=None, params=None, mask=None):
        #для общности в каждый узел передаем и потомков и параметры
        try:
            y_pred=pd.Series(np.nan, index= params['y'].index)

            y_pred.loc[mask]=params['params'][self.name]
            return y_pred
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_regr_const: name={0}, param={1}, ".format(self.name, params))
        return False

    
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_regr_const()
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



if __name__=='__main__':
    y=pd.Series(np.arange(0,1, 0.1), index=100*np.arange(0,1, 0.1))
    node=list_nom_class(value=1)
    rez=node.eval(params={'y':y}, mask=y>0.5)
    print(rez)

    node1=list_regr_const(name='const_0')
    rez=node1.eval(params={'y':y, 'params':{'const_0':101}}, mask=y<0.5)
    print(rez)

    

































