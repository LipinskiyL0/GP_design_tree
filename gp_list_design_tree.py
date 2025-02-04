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
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

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
        return f'{self.name}{self.koef}'
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
    
    def inf_gini(self, y):
        #Функция для определения информативности узла через коэффициент gini
        #https://education.yandex.ru/handbook/ml/article/reshayushchiye-derevya

        rez=y.value_counts()
        rez=rez/rez.sum()
        rez1=1-rez
        rez2=rez*rez1
        g=rez2.sum()
        return g
    
    def inf_mse(self, y):
        #Функция для определения информативности узла через коэффициент mse
        #хоть тут и нет в явном виде mse но если выполнить все преобразования, то
        #как раз получим дисперсию.
        #https://education.yandex.ru/handbook/ml/article/reshayushchiye-derevya
        if len(y)<=1:
            return 0
        g=y.var()
        return g
    def score(self, y_true, y_pred, metric='f1'):
        #единственная функция для доступа ко многим метрикам. Используется при обучении дерева и 
        #при оценке
        
        
        if metric=='f1':
            rez=f1_score(y_true, y_pred, average='macro')
        elif metric=='r2_score':
            rez=r2_score(y_true, y_pred)
        else:
            raise RuntimeError('Неизвестная метрика')
        return rez

class list_nom_class (list_tree_base):
    def __init__(self, value=0) -> None:
        self.name='nom_class'
        self.num_childs=0
        self.value=value
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
    def get_name_koef(self):
        return []
    def get_name(self):
        return f'{self.name}:{self.value}'
    def optim_koef(self, mask0,params=None):
        #учитывая особенности работы текущего узла подбор наилучшего значения осуществляется выбором мажоритарного класса
        #значений выхода
        y0=params['y'].loc[mask0]
        self.value=y0.value_counts().index[0]
        y_true=y0
        y_pred=np.full(len(y_true), self.value)
        rez=self.score( y_true, y_pred, metric='f1')
        return {'score':rez, 'inf_gate':False}
    
   
    
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
    def optim_koef(self, mask0,params=None):
        #учитывая особенности работы текущего узла подбор наилучшего значения осуществляется усреднением всех
        #значений выхода
        y0=params['y'].loc[mask0]
        name_coef=self.get_name_koef()
        self.koef[name_coef[0]]=np.mean(y0)
        y_true=y0
        y_pred=np.full(len(y_true), self.koef[name_coef[0]])
        rez=self.score( y_true, y_pred, metric='r2_score')

        return {'score':rez, 'inf_gate':False}
        

        
    
    
   
        
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
            mask=x<self.koef['p']
            return mask
        except:
            
            raise RuntimeError(f"Ошибка вычисления узла типа list_less: name={0}".format(self.name ))
        return False
    

    def get_name_koef(self):
        return ['p']
    def get_name(self):
        return f'{self.name}[{self.name_feature},{self.koef}]'
    def optim_koef(self, mask0,params=None, inf_name='gini' ):
        #учитывая особенности работы текущего узла можем реализовать подбор оптимального коэффициента
        #через перебор
        #находим нужный признак, находим уникальные значения признака и берем середины интервалов между значений
        if (self.name_feature in params['X'].columns)==False:
            raise RuntimeError(f'Отсутсвует признак {self.name_feature} в таблице данных' )
        unuc_val=params['X'][self.name_feature].loc[mask0].unique()
        unuc_val=np.sort(unuc_val)
        unuc_val1=(unuc_val[1:]+unuc_val[0:-1])/2
        koef_name=self.get_name_koef()[0]
        flag=0
        for p in unuc_val1:
            self.set_koef({koef_name:p})
            mask=self.eval(params)
            mask_inv=np.invert(mask)
            y0=params['y'].loc[mask0]
            y1=params['y'].loc[mask & mask0]
            y2=params['y'].loc[mask_inv & mask0]
            if inf_name=='gini':
                inf_y0=self.inf_gini(y0)
                inf_y1=self.inf_gini(y1)
                inf_y2=self.inf_gini(y2)
            elif inf_name=='mse':
                inf_y0=self.inf_mse(y0)
                inf_y1=self.inf_mse(y1)
                inf_y2=self.inf_mse(y2)
            inf_rez=inf_y0-(len(y1)*(inf_y1)/len(y0)+len(y2)*(inf_y2)/len(y0))
            if flag==0:
                best_inf_rez=inf_rez
                best_koef=p
                best_mask0=(mask & mask0).copy()
                best_mask1=(mask_inv & mask0).copy()
                flag=1
            elif best_inf_rez<inf_rez:
                best_inf_rez=inf_rez
                best_koef=p
                best_mask0=(mask & mask0).copy()
                best_mask1=(mask_inv & mask0).copy()
        self.set_koef({koef_name:best_koef})
        return {'score':False, 'inf_gate':best_inf_rez, 'mask0':best_mask0, 'mask1':best_mask1}
    
    
    


if __name__=='__main__':
    df=pd.Series(np.arange(1,0, -0.1), index=100*np.arange(0,1, 0.1))

    df.name='x1'
    df=df.reset_index()
    y=pd.Series(np.zeros(len(df), dtype=int))
    y.loc[df['x1']>0.5]=1

    node=list_regr_const( )

    mask=np.array([True]*len(df))

    rez=node.optim_koef(params={'X':df,'y':y}, mask0=mask)

    print(rez)
    print(df)
    

    

































