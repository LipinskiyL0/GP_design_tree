import pandas as pd
import numpy as np
from gp_tree import gp_tree
from gp_list_design_tree import *
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
            return False
        
        
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
        return koef
    
if __name__=='__main__':
    list_F=[]
    list_F.append(list_less(name_feature='x1' ))
    
    
    list_T=[]
    list_T.append(list_nom_class(value=1))
    list_T.append(list_nom_class(value=2))
    
    tree=gp_tree_design_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
                  limit_level=2)
    str_tree=tree.print_tree()
    print('\n')
    print(str_tree)

    koef=tree.get_koef()
    print(koef)
    i=10
    for k in koef.keys():
        koef[k]=i
        i=i+1
    print(tree.set_koef(koef))
    koef1=tree.get_koef()
    print(koef1)


