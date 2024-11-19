'''
28.01.2023 
В данном файле будет реализован типовой лист для дерева GP
В листе только реализуется вычисление текущего узла. Механика вычисления 
Всего дерева реализуется в другом файле

структура терминального узла:{
    self.name = имя узла
    self.value = значение для константного узла
    self.num_childs=0 т.к. у терминального узла нет потомков
    self.num_koef количество настраиваемых коэффициентов в узле
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
    get_name_koef функция возрващает типовые имена параметров

структура функционального узла:
    self.name - имя узла
    self.num_childs - количество потомков характерное для этой функции
    self.num_params количество параметров в узле
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
        self.num_koef=0
        self.koef={}
        if self.num_koef!=len(self.get_name_koef()):
            raise RuntimeError("Ошибка инициализации узла. Количество коэффициентов не совпадает с количеством имен")
    def eval(self,  params=None, mask=None):
        #для общности в каждый узел передаем и коэффициенты и параметры
        # создаем шаблон для выхода
        if len(self.koef)<self.num_koef:
            raise RuntimeError("Недостаточно коэффициентов в узле: name={0}".format(self.name))
        y_pred=pd.Series(np.nan, index= params['y'].index)

        y_pred.loc[mask]=self.value
        return y_pred
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_nom_class(value=self.value)
        rez.name=self.name
        rez.value=self.value
        rez.num_childs=self.num_childs
        rez.num_koef=self.num_koef
        rez.koef=self.koef.copy()
        return rez
    def get_name(self):
        return self.name+'_'+str(self.value)
    def get_name_koef(self):
        return []
    def set_koef(self, koef):
        names=self.get_name_koef()
        keys=koef.keys()
        for k in names:
            if (k in keys) ==False:
                raise RuntimeError("Ошибка установки параметра в узле")
        self.koef=koef.copy()
        return True
    
    def get_koef(self):
        return self.koef
    
#==============================================================================
class list_regr_const:
    def __init__(self) -> None:
        self.name='const'
        self.num_childs=0
        self.num_koef=1
        
        if self.num_koef!=len(self.get_name_koef()):
            raise RuntimeError("Ошибка инициализации узла. Количество коэффициентов не совпадает с количеством имен")

    def eval(self, params=None, mask=None):
        #для общности в каждый узел передаем и коэффициенты и параметры
        if len(self.koef) < self.num_koef:
            raise RuntimeError("Недостаточно коэффициентов в узле: name={0}".format(self.name))
        
        try:
            y_pred=pd.Series(np.nan, index= params['y'].index)

            y_pred.loc[mask]=self.koef['const']
            return y_pred
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_regr_const: name={0}, param={1}, ".format(self.name, params))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_regr_const()
        rez.name=self.name
        rez.num_childs=self.num_childs
        rez.num_koef=self.num_koef
        rez.koef=self.koef.copy()

        return rez
    
    def get_name(self):
        return self.name
    def get_name_koef(self):
        return ['const']
    def set_koef(self, koef):
        return True
    def get_koef(self):
        return self.koef
        
#==============================================================================
class list_less:
    def __init__(self, name_feature ) -> None:
        self.name='less'
        self.name_feature=name_feature
        self.num_childs=2
        self.num_koef=1
        if self.num_koef!=len(self.get_name_koef()):
            raise RuntimeError("Ошибка инициализации узла. Количество коэффициентов не совпадает с количеством имен")

    def eval(self, params=None):
        #для общности в каждый узел передаем и параметры и коэффициенты
        if len(self.koef)<self.num_koef:
            raise RuntimeError("Недостаточно коэффициентов в узле: name={0}".format(self.name))

        try:
            x=params['X'][self.name_feature]
            mask=x<koef['p']
            return mask
        except:
            
            raise RuntimeError(f"Ошибка вычисления узла типа list_less: name={0}".format(self.name ))
        return False
    def copy(self):
        #функция выполняет полную копию узла
        rez=list_less(self.name_feature)
        rez.name=self.name
        rez.name_feature=self.name_feature
        rez.num_childs=self.num_childs
        rez.num_koef=self.num_koef
        return rez
    
    def get_name(self):
        return f'{self.name}'
    def get_name_koef(self):
        return ['p']
    def set_koef(self, koef):
        return True
    def get_koef(self):
        return self.koef


if __name__=='__main__':
    y=pd.Series(np.arange(0,1, 0.1), index=100*np.arange(0,1, 0.1))
    node=list_nom_class(value=1)
    rez=node.eval(params={'y':y}, mask=y>0.5)
    node1=node.copy()
    print(node.get_name())
    print(rez)

    node=list_regr_const()
    rez=node.eval(params={'y':y, }, mask=y<0.5, koef={'const':101})
    node1=node.copy()
    print(node.get_name())
    print(rez)
    
    df=y.copy()
    df=df*1000
    df.name='x1'
    df=df.reset_index()
    node=list_less(name_feature='x1' )
    node1=node.copy()
    rez=node.eval(params={'X':df,'y':y},  koef={'p':101})
    print(node.get_name())
    print(rez)
    print(df)
    

    

































