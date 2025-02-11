'''
В данном классе реализован узел дерева со всей необходимой механникой. 
В классе решаются две задачи: 
1. Вычислительная часть. За нее отвечает объект одного из типов, описанных в gp_list_design_tree.py. Вся вычислительная
    часть реализуется в gp_list_design_tree.py, а здесть только вызывается. 
2. Навигация по дереву и эволюционные операторы. Вот эта часть описывается в классе gp_tree_design_tree.

основные объекты: 
self.list - объект типа gp_list_design_tree.py в котором реализована вычислительная часть текущего узла
self.level - уровень текущего узла начиная с нуля
self.nom_list - номер текущего узла. В основном используется при сопоставлении деревьев при рекомбинации, а также 
                для задания уникального имени числовому параметру узла при оптимизации дерева
self.childs - список узлов этого же типа gp_tree_design_tree представляющих собой потомков для текущего узла,
                который для них является родительским. Потомки в свою очередь содержат в себе своих потомков и т.д. 
self.num_childs - количество потомков, т.е. длина массива self.childs. Ну это некоторая дань C++

Основные методы и их параметры
                
def __init__(self, list_T=None, list_F=None, level=0, nom_list='1', type_ini='full',
                 limit_level=2, childs=[], cur_list=None, params=None) -> None:
            list_T - список терминальных узлов
            list_F - список функциональных узлов
            level -  уровень текущего узла в дереве начиная с нуля
            nom_list - глобальный номер узла в дереве начиная с 1. Номера потомков: 1.1 и 1.2. Номера их потомков
                        1.1.1, 1.1.2, 1.2.1 и 1.2.2 и т.д. 
            type_ini - тип инициализации дерева. Есть два стандартных способа роста: full - полное, nofull - не полное
                    кроме тоого есть: null - инициализация пустого узла, manual - ручная точечная инициализация узла,
                    LearnID3 - жадная стратегия роста дерева. 
def get_koef(self) - Функция возвращает коэффициенты текущего узла и всего поддерева целиком в виде словаря
def set_koef(self, koef) - Функция устанавливает коэффициенты в текущий узел из словаря koef
                            определяем ключи, которые должны быть в данном узле
def eval(self, params, mask=[]) - Функция производит обход дерева и его вычисление 
                                  mask - массив со значениями True/False. Вычисление узла происходит только для тех
                                  объектов выборки для которых mask==True
def predict(self, X): - вычисление дерева. Надстройка над функцией eval для того, что бы соответствовать терминам
                        sklearn и что бы удобнее пользоваться было
def score(self, X, y, metric='f1'): - единственная функция для доступа ко многим метрикам. Используется при обучении дерева и 
                                    при оценке
def loss(self, x0, X, y, metric, list_keys): функция для обучения дерева. Определяет функцию потерь. основывается на стандартных метриках
                                            преобразует метрики в функцию потерь, т.е. чем меньше значение тем лучше. 
                                            x0 - массив количественных коэффициентов узлов, для которых вычисляется функция.
                                                эти коэффициенты устанавливаются на дерево после чего проихсодт оценка точности
                                            X - входы
                                            y - выходы
                                            metric - метрика потерь
                                            list_keys - список имен коэффициентов, что бы знать какой коэффициент в какой узел 
                                                        устанавливать
def fit(self, X, y, metric='f1', restart=True, method='Nelder-Mead', iterations=100, inf_name='gini'):
        #обучение дерева - настройка количественных коэффициентов. Функция приеняет алгорит оптимизации и минимизирует
        loss функцию
        X - входы
        y - выходы
        metric - метрика потерь
        restart - флаг рестарта - испльзуется только в методе Нелдера-Мида. Что бы поиск стартовал с текущих значений
                по умолчанию коэффициенты обучаются с нуля.
        method - метод оптимизации. Включая самооптимизацию, которая подбирает наилучшие значения за счет 
                функций локальной самооптимизации узлов def optim_koef(self, mask0,params=None): классов в 
                gp_list_design_tree.py
        iterations - параметр количества итерация для итерационных методов
        inf_name - параметр метрики информативности разбиения, который используется в локальнй самооптимизации узла
def LearnID3(self, params): - процедура в которой реализована стратегия жадного роста дерева
def get_optim_list_T(self, y, list_T, mask): Функция ищет простым перебором наилучшее решение из
                                             терминальных листов используется в LearnID3

def get_optim_list_F(self, X, y, list_F,mask, inf_name): - Функция ищет простым перебором наилучшее решение из
                                             функциональных листов используется в LearnID3
                                            

'''

import pandas as pd
import numpy as np
from gp_tree import gp_tree
from gp_list_design_tree import *
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize
from DivClass import DivClass
from bcolors import bcolors
from sklearn.datasets import load_diabetes

class gp_tree_design_tree(gp_tree):
    def __init__(self, list_T=None, list_F=None, level=0, nom_list='1', type_ini='full',
                 limit_level=2, childs=[], cur_list=None, params=None) -> None:
        
        #происходит инициализация дерева рекурсивным способом
        if type_ini=='full':
            #инициализируем дерево методом полного роста
            if level<limit_level:
                #если не достигли еще глубины инциализируем функциональным узлом
                i=np.random.randint(len(list_F))
                self.list=list_F[i].copy()
            else:
                i=np.random.randint(len(list_T))
                self.list=list_T[i].copy()
        elif type_ini=='nofull':
            #инициируем дерево методом не полного роста
            if level<limit_level:
                #если не достигли еще глубины инциализируем либо функциональным узлом,
                #либо терминальным
                if np.random.rand()<0.5:
                    #инициализируем функциональным узлом
                    i=np.random.randint(len(list_F))
                    self.list=list_F[i].copy()
                else:
                    #инициализируем терминальным узлом
                    i=np.random.randint(len(list_T))
                    self.list=list_T[i].copy()
            else:
                #если уровень предельный, то по любому инициируем терминальным узлом
                i=np.random.randint(len(list_T))
                self.list=list_T[i].copy()
            
        elif type_ini=='null':
            #инициируем только один  пустой узел без вызова дочерних узлов
            self.level=level 
            self.nom_list=nom_list
            self.list=None
            self.num_childs=0
            self.childs=[]
            return
        elif type_ini=='manual':
            #инициируем узел в ручную. При этом потомки передаются через переменную childs, а 
            #вычислительная часть узла через переменную cur_list
            #номерация узлов тоже выполняется вручную
            self.level=level 
            self.nom_list=nom_list
            self.list=cur_list
            self.num_childs=len(childs)
            self.childs=childs
            self.num_koef=cur_list.num_koef
            return

        elif type_ini=='LearnID3':
            if pd.isnull(params)==True:
                raise RuntimeError("В методе роста LearnID3 отсутствует выборка")
            if ('X'in params) == False:
                raise RuntimeError("В методе роста LearnID3 с параметрами не передается выборка X")
            if type(params['X']) != pd.core.frame.DataFrame:
                raise RuntimeError("В методе роста LearnID3 выборка X не формата DataFrame")
            
            if ('y'in params) == False:
                raise RuntimeError("В методе роста LearnID3 с параметрами не передается выборка y")
            if type(params['y']) != pd.core.series.Series:
                raise RuntimeError("В методе роста LearnID3 выборка y не формата pd.Series")
            

            if level == 0:
                params['list_T']=list_T
            if level == 0:
                if params['n_features']<0:
                    params['list_F']=list_F.copy()
                else:
                    list_F_temp=pd.Series(list_F)
                    list_F_temp=list_F_temp.sample(params['n_features'])
                    params['list_F']=list(list_F_temp)
            rez=self.LearnID3(params)
            self.list=rez['list']
            self.level=level
            self.nom_list=str(nom_list)
            self.childs=[]
            self.num_childs=self.list.num_childs
            
            if self.num_childs==0:
                # достигли терминального узла. возвращаемся
                return
            if self.num_childs!=2:
                raise RuntimeError('количество потомков не равно двум. LearnID3 на это не рассчитана')
            params0=params.copy()
            params0['mask']=rez['mask0']

            params1=params.copy()
            params1['mask']=rez['mask1']
            Params=[params0, params1]

            #иначе продолжаем
            for i, p in enumerate(Params):
                сhild= gp_tree_design_tree(list_T=list_T, list_F=list_F, level=level+1, nom_list=nom_list+'.'+str(i+1),
                            type_ini=type_ini, limit_level=limit_level, params=p)
                self.childs.append(сhild)
            
            return


        else:
            raise RuntimeError("Ошибка определения метода инициализации дерева {0}".format(type_ini))
            
        
        self.level=level
        self.nom_list=str(nom_list)
        self.childs=[]
        self.num_childs=self.list.num_childs
        
        for i in range(self.num_childs):
            сhild= gp_tree_design_tree(list_T=list_T, list_F=list_F, level=level+1, nom_list=nom_list+'.'+str(i+1),
                          type_ini=type_ini, limit_level=limit_level)
            self.childs.append(сhild)
        
        return

    def get_koef(self):
        # Функция возвращает коэффициенты текущего узла и всего поддерева целиком
        koef0=self.list.get_koef()
        koef={}
        for k in koef0.keys():
            koef[f'{self.nom_list}_{k}']=koef0[k]
        for ch in self.childs:
            koef.update(ch.get_koef())
        return koef

    def set_koef(self, koef):
        # Функция устанавливает коэффициенты в текущий узел из словаря koef
        # определяем ключи, которые должны быть в данном узле
        keys0=self.list.get_name_koef()
        # находим нужные параметры с учетом номера узла (глобальный ключ)
        koef_list={}
        for k in keys0:
            kk=f'{self.nom_list}_{k}'
            if (kk in koef)==False:
                raise RuntimeError(f'Отсутствует нужный параметр в узле: {self.nom_list}, параметр: {kk}')
            koef_list[k]=koef[kk]
        
        fl=self.list.set_koef(koef_list)

        for ch in self.childs:
            ch.set_koef(koef)
            
        if fl==False:
            RuntimeError(f'ошибка установки параметров в узле: {self.nom_list} ')
        return True

#--------------------------------------------------------------------------            
    def eval(self, params, mask=[]):
        #Функция производит обход дерева и его вычисление 
        if len(mask)==0:
            mask=np.array([True]*len(params['y']))
            
        #вычисление дерева
        if len(self.childs)==0:
            return self.list.eval(params=params, mask=mask)
        elif len(self.childs)!=2:
            raise RuntimeError("Количество потомков не равно двум в узле номер: {0}".format(self.nom_list ))
        else:
            try:
                mask0=self.list.eval(params=params)
                mask1=np.invert(mask0)
                if len(mask)!=0:
                    mask0=(mask0 & mask)
                    mask1=(mask1 & mask)
            except:
                raise RuntimeError("Ошибка вычисления узла номер: {0}".format(self.nom_list ))

            #вычисляем левую ветку
            rez0=self.childs[0].eval(params=params, mask=mask0)
            rez1=self.childs[1].eval(params=params, mask=mask1)
            if type(rez0)==pd.core.series.Series:
                rez0.loc[mask1]=rez1.loc[mask1]
            elif type(rez0)==np.ndarray:
                rez0[mask1]=rez1[mask1]
            

            # childs=[]
            # for i in range(len(self.childs)):
            #     childs.append(self.childs[i].eval(params=params))
        
        return rez0
    #--------------------------------------------------------------------------            
    def predict(self, X):
        #вычисление дерева. Надстройка над функцией eval для того, что бы соответствовать терминам
        #sklearn и что бы удобнее пользоваться было
        if type(X)!=pd.core.frame.DataFrame:
            raise RuntimeError('Входящее значение должно быть формата DataFrame')
        y=np.zeros(len(X))
        y=pd.Series(y, index=X.index)
        params={'X':X, 'y':y}
        y_pred=self.eval(params)
        if np.sum(pd.isnull(y_pred))>0:
            print(bcolors.FAIL + f'Выход дерева содержит пропуски'+ bcolors.ENDC)
            print(bcolors.FAIL + self.print_tree()+ bcolors.ENDC)
        return y_pred
 #--------------------------------------------------------------------------               
    def score(self, X, y, metric='f1'):
        #единственная функция для доступа ко многим метрикам. Используется при обучении дерева и 
        #при оценке
        y_pred=self.predict(X)
        
        if metric=='f1':
            rez=f1_score(y, y_pred, average='macro')
        elif metric=='mse':
            rez=mse(y, y_pred)
        elif metric=='r2_score':
            rez=r2_score(y, y_pred)
        else:
            raise RuntimeError('Неизвестная метрика')
        return rez
#--------------------------------------------------------------------------               
    def loss(self, x0, X, y, metric, list_keys):
        #функция для обучения дерева. Определяет функцию потерь. основывается на стандартных метриках
        #преобразует метрики в функцию потерь, т.е. чем меньше значение тем лучше. 
        if (x0 is None)==False:
            koef={}
            for key, x in zip(list_keys, x0):
                koef[key]=x
                
            self.set_koef(koef)
        e=self.score(X=X, y=y, metric=metric)
        if metric=='f1':
            # переворачиваем для минимизации
            e=1-e 
        elif metric=='r2_score':
            e=1-e
        return e
#--------------------------------------------------------------------------            
    def fit(self, X, y, metric='f1', restart=True, method='Nelder-Mead', iterations=100, inf_name='gini'):
        #обучение дерева - настройка количественных коэффициентов
        koef=self.get_koef()
        list_keys=[]
        list_x=[]
        for k in koef:
            list_keys.append(k)
            if restart:
                list_x.append(np.random.rand())
            else:
                list_x.append(koef[k])
        x0=np.array(list_x)  
        args=(X, y, metric, list_keys)
        if method=='Nelder-Mead':
            res = minimize(self.loss, x0, method=method, options={'gtol': 1e-6, 'disp': True},args=args)
            x1=res.x
        elif method=='DE':
            Div=DivClass()
            Div.inicialization(FitFunct=self.loss, Min=np.zeros(len(x0)), Max=np.ones(len(x0)), args=args, n_ind=100)
            x1=Div.opt(n_iter=iterations, args=args)  
        elif method=='self_optimization':
            params={'X':X, 'y':y, 'inf_name':inf_name,'mask':np.full(len(y), True) }
            self.self_optimization(params)
            koef=self.get_koef()
            x1=[]
            for k in koef:
                x1.append(koef[k])

        else:
            raise RuntimeError(f'неизвестный метод оптимизации: {method}')
 

        # устанавливаем наилучшее решение и возвращаем точность
        e=self.loss(x1, X, y, metric, list_keys)
        return e
    def self_optimization(self, params):
        y=params['y']
        X=params['X']
        inf_name=params['inf_name']

        mask=params['mask']
        if self.num_childs==0:
            # терминальный узел
            rez=self.list.optim_koef(params={'y':y}, mask0=mask)
        elif self.num_childs==2:
            # функциональный узел с двумя потомками
            rez=self.list.optim_koef(params={'X':X,'y':y}, mask0=mask, inf_name=inf_name )
            # if rez['fl_success']==False:
            #     print(f'Ошибка оптимизации узла в процедуре самооптимизации: номер узла {self.nom_list}')

            params0=params.copy()
            params0['mask']=rez['mask0']

            params1=params.copy()
            params1['mask']=rez['mask1']
            Params=[params0, params1]

           
            for i, p in enumerate(Params):
                self.childs[i].self_optimization(p)
        else:
            raise RuntimeError(f'Ошибка в процедуре самооптимизации, неизвестное количество потомков: {self.num_childs}')
        return
    
    def LearnID3(self, params):
        # В данной функции реализован метод выращивания дерева через процедуру ID3
        X=params['X']
        y=params['y']
        list_F=params['list_F']
        list_T=params['list_T']
        mask=params['mask']
        epsilon=params['epsilon']
        num_samples=params['num_samples']
        inf_name=params['inf_name']

        if mask is None:
            mask=np.full(len(y), True)
        
        # Если все объекты лежат в одном классе или равны одному числу, то делаем конечый (терминальный) узел
        # Если количество точек в узле меньше порогового
        if (y.loc[mask].var()<=epsilon) | (sum(mask)<num_samples):
            rez=self.get_optim_list_T(y, list_T, mask)
            
        # Находим предикат с максимальным информационным выигрышем
        else:
            rez=self.get_optim_list_F(X, y, list_F, mask, inf_name)
            # если поиск закончился неудачно, то ставим терминальный узел
            if rez['fl_success']==False:
                rez=self.get_optim_list_T(y, list_T, mask)
            # если одна из веток содержит объектов меньше чем num_samples ставим терминальный узел
            elif (sum(rez['mask0'])< num_samples) | (sum(rez['mask1'])< num_samples):
                rez=self.get_optim_list_T(y, list_T, mask)

        return rez
    
    def get_optim_list_T(self, y, list_T, mask):
        #Функция ищет простым перебором наилучшее решение из терминальных листов
        flag=False
        # mask=np.full(len(y), True)
        for li in list_T:
            rez=li.optim_koef(params={'y':y}, mask0=mask)
            if flag==False:
                best_score=rez['score']
                best_list=li.copy()
                flag=True
            elif best_score<rez['score']:
                best_score=rez['score']
                best_list=li.copy()
        return {'score':best_score, 'list':best_list, 'mask0':None, 'mask1':None}
    
    def get_optim_list_F(self, X, y, list_F,mask, inf_name):
        #Функция ищет простым перебором наилучшее решение из функциональных листов
        flag=False
        fl_success=False
        # mask=np.full(len(y), True)
        for li in list_F:
            rez=li.optim_koef(params={'X':X,'y':y}, mask0=mask, inf_name=inf_name )
            if rez['fl_success']==False:
                continue
            if flag==False:
                best_inf=rez['inf_gate']
                best_list=li.copy()
                best_mask0=rez['mask0']
                best_mask1=rez['mask1']
                flag=True
                fl_success=True
            elif best_inf<rez['inf_gate']:
                best_inf=rez['inf_gate']
                best_list=li.copy()
                best_mask0=rez['mask0']
                best_mask1=rez['mask1']
        
        if fl_success==False:
            return {'fl_success': False}
        return {'inf_gate':best_inf, 'list':best_list, 'mask0':best_mask0, 'mask1':best_mask1, 'fl_success': True}

    
if __name__=='__main__':
    

    data = load_diabetes()
    X=data.data
    X=pd.DataFrame(X, columns=data.feature_names)
    y=data.target
    y=pd.Series(y, name='y')
    # params={'X':X, 'y':y,'method':'DE','score_metric':'f1','iterations':20   }
    params={'X':X, 'y':y,'mask':None,'epsilon':1e-10, 'num_samples':4,'inf_name':'mse', 'list_F':None, 'list_T':None,
            'n_features':1,'method':'self_optimization','score_metric':'mse','iterations':50  }
    list_T=[]
    list_T.append(list_regr_const())

    list_F=[]
    for c in X.columns:
        list_F.append(list_less(name_feature=c ))

    tree=gp_tree_design_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='LearnID3',
                    limit_level=2, params=params)
    
    str_tree=tree.print_tree()
    print(str_tree)
    e=tree.score(X=X, y=y, metric='r2_score')
    print(f'точность работы на исходном дереве: {e}')
    # Проверяем функцию fit
    e=tree.fit(X=X,y=y, method='self_optimization',metric='mse', inf_name='mse')
    print(f'loss: {e}')
    e=tree.score(X=X, y=y, metric='r2_score')
    print(f'точность работы на исходном дереве: {e}')
    y_pred=tree.predict(X=X)
    y_pred.name='y_pred'
    print('Сравниваем "реальны" и вычисленный выходы')

    print(pd.concat([X, y,y_pred], axis=1))
    mse_score=tree.score(X=X, y=y, metric='r2_score')
    print()
    print(mse_score)
    print(tree.print_tree())   


