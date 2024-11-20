'''
28.01.2023 
В данном файле будет реализован типовой лист для дерева GP
В листе только реализуется вычисление текущего узла. Механика вычисления 
Всего дерева реализуется в другом файле

структура базового узла:{
    self.name = имя узла
    
    self.num_childs=0 т.к. у терминального узла нет потомков
    self.num_koef количество настраиваемых коэффициентов в узле
    self.koef - словарь коэффициентов
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
    set_koef функция устанавливает коэффициенты в узел
    get_koef функция копирует коэффициенты из узла


'''

import numpy as np
import  pandas as pd

class list_tree_base:
    def __init__(self) -> None:
        self.name=''
        self.num_childs=0
        
        keys=self.get_name_koef()
        self.koef={}
        for k in keys:
            self.koef[k]=0
    def eval(self,  params=None, mask=None):
        keys=self.get_name_koef()
        for k in keys:
            if (k in self.koef)==False:
                raise RuntimeError("Недстаточно параметров в узле для вычисления")
            
        return False
    def copy(self):
        #функция выполняет полную копию узла
        cl=type(self)
        rez=cl()
        # получаем список атрибутов класса
        attr={k: v for k, v in self.__dict__.items() if not (k.startswith('__') and k.endswith('__'))}
        for key in attr:
            setattr(rez, key, attr[key])
        return rez
    
    def get_name(self):
        return f'{self.name}{self.koef})'
    def get_name_koef(self):
        return []
    def set_koef(self, koef):
        names=self.get_name_koef()
        keys=koef.keys()
        for k in names:
            if (k in keys) ==False:
                raise RuntimeError("Ошибка установки параметра в узле")
        for k in keys:
            if (k in names) ==False:
                raise RuntimeError("Ошибка установки параметра в узле")
        self.koef=koef.copy()
        return True
    
    def get_koef(self):
        return self.koef

class list_nom_class (list_tree_base):
    def __init__(self, value=0) -> None:
        self.name='nom_class'
        self.value=value
        self.num_childs=0
        keys=self.get_name_koef()
        self.koef={}
        for k in keys:
            self.koef[k]=0

    def eval(self,  params=None, mask=None):
        #для общности в каждый узел передаем и коэффициенты и параметры
        # создаем шаблон для выхода
        keys=self.get_name_koef()
        for k in keys:
            if (k in self.koef)==False:
                raise RuntimeError("Недстаточно параметров в узле для вычисления")
        y_pred=pd.Series(np.nan, index= params['y'].index)
        y_pred.loc[mask]=self.value
        return y_pred
    
   
    
#==============================================================================
class list_regr_const(list_tree_base):
    def __init__(self) -> None:
        self.name='const'
        self.num_childs=0
        
        keys=self.get_name_koef()
        self.koef={}
        for k in keys:
            self.koef[k]=0
        
    def eval(self, params=None, mask=None):
        #для общности в каждый узел передаем и коэффициенты и параметры
        keys=self.get_name_koef()
        for k in keys:
            if (k in self.koef)==False:
                raise RuntimeError("Недстаточно параметров в узле для вычисления")
                    
        try:
            y_pred=pd.Series(np.nan, index= params['y'].index)

            y_pred.loc[mask]=self.koef['const']
            return y_pred
        except:
            
            raise RuntimeError("Ошибка вычисления узла типа list_regr_const: name={0}, param={1}, ".format(self.name, params))
        return False
    
    def get_name_koef(self):
        return ['const']
   
        
#==============================================================================
class list_less(list_tree_base):
    def __init__(self, name_feature='' ) -> None:
        self.name='less'
        self.name_feature=name_feature
        self.num_childs=2
        
        keys=self.get_name_koef()
        self.koef={}
        for k in keys:
            self.koef[k]=0
        
    def eval(self, params=None):
        #для общности в каждый узел передаем и параметры и коэффициенты
        keys=self.get_name_koef()
        for k in keys:
            if (k in self.koef)==False:
                raise RuntimeError("Недстаточно параметров в узле для вычисления")
            
        try:
            x=params['X'][self.name_feature]
            mask=x<koef['p']
            return mask
        except:
            
            raise RuntimeError(f"Ошибка вычисления узла типа list_less: name={0}".format(self.name ))
        return False
    

    def get_name_koef(self):
        return ['p']
    


if __name__=='__main__':
    l=list_tree_base()
    print(f'get_name: {l.get_name()}')
    l.name='test'
    l.num_childs=2
    l.num_koef=3
    l.koef={'p1':-1, 'p2':-2}
    
    l1=l.copy()
    print(f'get_name: {l1.get_name()}')
    print(l1.get_name_koef())
    print(l1.get_koef())
    

    y=pd.Series(np.arange(0,1, 0.1), index=100*np.arange(0,1, 0.1))
    node=list_nom_class(value=1)
    rez=node.eval(params={'y':y}, mask=y>0.5)
    node1=node.copy()
    print(node.get_name())
    print(rez)

    node=list_regr_const()
    koef={'const':101}
    node.set_koef(koef)
    rez=node.eval(params={'y':y, }, mask=y<0.5)
    node1=node.copy()
    print(node.get_name())
    print(rez)
    
    df=y.copy()
    df=df*1000
    df.name='x1'
    df=df.reset_index()
    node=list_less(name_feature='x1' )
    node1=node.copy()
    koef={'p':101}
    node.set_koef(koef)
    rez=node.eval(params={'X':df,'y':y}  )
    print(node.get_name())
    print(rez)
    print(df)
    

    

































