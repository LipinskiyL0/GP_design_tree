import pandas as pd
import numpy as np
from gp_tree import gp_tree
from gp_list_design_tree import *
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize
from DivClass import DivClass

class gp_tree_design_tree(gp_tree):
    def __init__(self, list_T=None, list_F=None, level=0, nom_list='1', type_ini='full',
                 limit_level=2, childs=[], cur_list=None) -> None:
        
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
        if type(X)!=pd.core.frame.DataFrame:
            raise RuntimeError('Входящее значение должно быть формата DataFrame')
        y=np.zeros(len(X))
        y=pd.Series(y, index=X.index)
        params={'X':X, 'y':y}
        y_pred=self.eval(params)
        return y_pred
 #--------------------------------------------------------------------------               
    def score(self, X, y, metric='f1'):
        y_pred=self.predict(X)
        if metric=='f1':
            rez=f1_score(y, y_pred, average='macro')
        elif metric=='mse':
            rez=mse(y, y_pred)
        else:
            raise RuntimeError('Неизвестная метрика')
        return rez
#--------------------------------------------------------------------------               
    def loss(self, x0, X, y, metric, list_keys):
        # X=args[0]
        # y=args[1]
        # metric=args[2]
        # list_keys=args[3]

        koef={}
        for key, x in zip(list_keys, x0):
            koef[key]=x
            
        self.set_koef(koef)
        e=self.score(X=X, y=y, metric=metric)
        if metric=='f1':
            # переворачиваем для минимизации
            e=1-e 
        return e
#--------------------------------------------------------------------------            
    def fit(self, X, y, metric='f1', restart=True, method='Nelder-Mead'):
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
            Div.inicialization(FitFunct=self.loss, Min=np.zeros(len(x0)), Max=np.ones(len(x0)), args=args)
            x1=Div.opt(n_iter=100, args=args)   

        # устанавливаем наилучшее решение и возвращаем точность
        e=self.loss(x1, X, y, metric, list_keys)
        return e
    
if __name__=='__main__':
    list_F=[]
    
    list_F.append(list_less(name_feature='x1' ))
    list_F.append(list_less(name_feature='x2' ))
    list_T=[]
    list_T.append(list_regr_const())

    tree=gp_tree_design_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
                    limit_level=2)
    str_tree=tree.print_tree()

    # Проверяем функцию fit
    df=np.random.rand(10,2)
    df=pd.DataFrame(df, columns=['x1', 'x2'])
    df['y']=2*df['x1']+3*df['x2']

    e=tree.fit(X=df,y=df['y'], method='DE',metric='mse')
    print(f'loss: {e}')
    y_pred=tree.predict(X=df)
    y_pred.name='y_pred'
    print('Сравниваем "реальны" и вычисленный выходы')

    print(pd.concat([df,y_pred], axis=1))
    mse_score=tree.score(X=df, y=df['y'], metric='mse')
    print()
    print(mse_score)
    print(tree.print_tree())   


