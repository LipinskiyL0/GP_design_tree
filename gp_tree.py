'''

28.01.2023 
Класс реализует дерево для реализации расчетов генетического программирования

    

'''

import numpy as np
import pandas as pd
from gp_list import *

class gp_tree:
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
            return
            
        else:
            raise RuntimeError("Ошибка определения метода инициализации дерева {0}".format(type_ini))
            return False
        
        
        self.level=level
        self.nom_list=str(nom_list)
        self.childs=[]
        self.num_childs=self.list.num_childs
        for i in range(self.num_childs):
            сhild=gp_tree(list_T=list_T, list_F=list_F, level=level+1, nom_list=nom_list+'.'+str(i+1),
                          type_ini=type_ini, limit_level=limit_level)
            self.childs.append(сhild)
        
        return
    #--------------------------------------------------------------------------
    def print_tree(self):
        #отображение дерева в строку
        if len(self.childs)==0:
            return self.list.get_name()
        else:
            rez=''
            rez+=self.list.get_name()+'('
            for i in range(len(self.childs)):
                rez+=self.childs[i].print_tree()
                
                if i!=len(self.childs)-1:
                    rez+=', '
                
            rez+=')'
        return rez
    #--------------------------------------------------------------------------            
    def eval(self, params):
        #отображение дерева в строку
        if len(self.childs)==0:
            return self.list.eval(params=params)
        else:
            childs=[]
            for i in range(len(self.childs)):
                childs.append(self.childs[i].eval(params=params))
        try:
            rez=self.list.eval(childs=childs, params=params)
        except:
            raise RuntimeError("Ошибка вычисления узла номер: {0}".format(self.nom_list ))
        return rez
    #--------------------------------------------------------------------------
    def copy(self):
        #copy текущего узла и всего поддерева
        
        tree_list=gp_tree(level=self.level, nom_list=self.nom_list,
                      type_ini='null', limit_level=0)
        tree_list.list=self.list.copy()
            
        if len(self.childs)!=0:
            childs=[]
            for i in range(len(self.childs)):
                childs.append(self.childs[i].copy())
            tree_list.childs=childs
            tree_list.num_childs=len(childs)
        return tree_list
    #--------------------------------------------------------------------------
    def recombination(self, gp_tree, old ):
        #для текущего узла и его подузлов находим узел с номером old
        #заменяем его содержимое содержимым  gp_tree включая все поддерево. 
        #затем обновляем нумерацию
        if self.nom_list==old:
            #нашли нужный лист 
            #заменяем функционал или терминал и поддерево потомков
            self.list=gp_tree.list.copy()
            self.num_childs=gp_tree.num_childs
            self.childs=gp_tree.childs
            if self.num_childs!=len(self.childs):
                raise RuntimeError("Ошибка рекомбинации поддерева в узле {0} количество потомков не сходится".format(self.nom_list))
                return False
            for i in range(len(self.childs)):
                self.childs[i].update_numper(level=self.level+1, nom_list=self.nom_list+'.'+str(i+1))
        else:
            for i in range(len(self.childs)):
                self.childs[i].recombination(gp_tree, old )
                
        return True
    #--------------------------------------------------------------------------
    def update_numper(self, level, nom_list ):
        #для текущего узла и его подузлов обновляем нумерацию
        #используется в recombination
        self.level=level
        self.nom_list=str(nom_list)
        for i in range(len(self.childs)):
            self.childs[i].update_numper(level=level+1, nom_list=nom_list+'.'+str(i+1))
        return True
    #--------------------------------------------------------------------------
    def get_tree_number(self):
        #отображение номеров узлов для проверки корректности процедуры рекомбинации
        if len(self.childs)==0:
            return self.nom_list
        else:
            rez=''
            rez+=self.nom_list+'('
            for i in range(len(self.childs)):
                rez+=self.childs[i].get_tree_number()
                
                if i!=len(self.childs)-1:
                    rez+=', '
                
            rez+=')'
        return rez
    #--------------------------------------------------------------------------
    def get_node(self, nom):
        #возвращает узел по номеру
        
        #если нашли нужный узел, то его и возвращаем
        if self.nom_list==nom:
            return self
        
        #если мы дошли до терминального узла, то возвращаем False
        if self.num_childs==0:
            return False
        
        #если это функциональный узел, то перебираем потомков и проверяем кто из
        #них возвращает не False. Если все потомки возвращают False, то 
        #то возвращаем False
        for i in range(len(self.childs)):
            node=self.childs[i].get_node(nom)
            if node!=False:
                return node
        return False
    #--------------------------------------------------------------------------
    def mutation(self, p, mas_for_mut):
        #мутирует текущий узел на узел с равным количеством потомков 
        #с вероятностью p. mas_for_mut - массив с перечнем узлов.
        #формат массива: [сам узел,название узла, количество потомков]
        #для терминальных узлов указано 0 потомков
        
        #алгоритм:
        #разыгрываем вероятность для текущего узла. Если событие случилось, то
        #выбираем из списка узлы с соответствующим количеством потомков и 
        #выбираем случайно на какой узел заменить. 
        #далее, неважно случилось событие или нет вызываем дочерние узлы
        
        
    
        #если нашли нужный узел, то его и возвращаем
        if np.random.rand()<=p:
            #событие случилось
            mas_for_mut_n=mas_for_mut[mas_for_mut['num_childs']==self.list.num_childs]
            if len(mas_for_mut_n)!=0:
                list_new=mas_for_mut_n['list'].iloc[np.random.randint(len(mas_for_mut_n))]
                self.list=list_new.copy()
        
        #если мы дошли до терминального узла, то возвращаемся
        if self.num_childs==0:
            return
        
        #если это функциональный узел, то перебираем потомков и вызываем мутацию
        #в каждом потомке
        for i in range(len(self.childs)):
            node=self.childs[i].mutation(p, mas_for_mut)
            
        return
    def get_num_node(self):
        #считаем количество вершин
        #если мы дошли до терминального узла, то возвращаем 1
        if self.num_childs==0:
            return 1
        #если это функциональный узел, то считаем количество вершин в поддереве
        s=0
        for i in range(len(self.childs)):
            s+=self.childs[i].get_num_node()
        return s
                
        
        

                            
    
if __name__=='__main__':
    list_F=[]
    list_F.append(list_sum(num_childs=1))
    list_F.append(list_sum(num_childs=2))
    list_F.append(list_sum(num_childs=3))
    
    list_F.append(list_prod(num_childs=1))
    list_F.append(list_prod(num_childs=2))
    list_F.append(list_prod(num_childs=3))
    
    list_F.append(list_sin())
    
    list_T=[]
    list_T.append(list_const(0))
    list_T.append(list_const(1))
    list_T.append(list_const(3.14))
    list_T.append(list_const(2.71))
    list_T.append(list_variable(name='x1'))
    list_T.append(list_variable(name='x2'))
    list_T.append(list_variable(name='x3'))
    list_T.append(list_variable(name='x4'))
    
    #==========================================================================
    # tree=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
    #               limit_level=2)
    # # проверка корректности вычисления
    # str_tree=tree.print_tree()
    # print('\n')
    # print(str_tree)
    # print(tree.get_tree_number())
    # params={'x1':1,'x2':2, 'x3':3, 'x4':4 }
    # rez=tree.eval(params)
    # print(rez)
    # copy_tree=tree.copy()
    
    # str_copy_tree=copy_tree.print_tree()
    # print(str_copy_tree)
    # print(copy_tree.get_tree_number())
    # rez_copy=copy_tree.eval(params)
    # print(rez_copy)
    #==========================================================================
    #проверка корректности рекомбинации 
    # tree1=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
    #               limit_level=2)
    # tree2=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
    #               limit_level=2)
    # print('первое дерево')
    # print(tree1.print_tree())
    # print('\n второе дерево')
    # print(tree2.print_tree())
    
    # tree1.recombination(tree2, '1.1.1')
    # print("дерево 3\n",tree1.print_tree())
    
    # print('\n номерация \n', tree1.get_tree_number())
    
    #==========================================================================
    
    #проверка корректности рекомбинации  2
    # tree1=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
    #              limit_level=2)
    # tree2=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
    #              limit_level=2)
    
    # print('первое дерево')
    # print(tree1.print_tree())
    # print('\n второе дерево')
    # print(tree2.print_tree())
    
    # node=tree2.get_node('1.1')
    # if node!=False:
    #     tree1.recombination(node, '1.1')
    # print("дерево 3\n",tree1.print_tree())
    
    print('\n номерация \n', tree1.get_tree_number())
    
    #==========================================================================
    #проверка корректности мутации
    # tree1=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
    #               limit_level=10)
    # mas_for_mut=[]
    # for l in list_F+list_T:
    #     mas_for_mut.append([l,l.get_name(), l.num_childs])
    
    # mas_for_mut=pd.DataFrame(mas_for_mut, columns=['list', 'list_name', 'num_childs'])
    # print(tree1.print_tree())
    # tree1.mutation(0.3, mas_for_mut)
    # print(tree1.print_tree())
    