# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 21:09:15 2023

в этом файле реализуются рекомбинации двух видов для генетического программирования
общая формула селекции: на входе два дерева - родителя на выходе дерево - потомок

@author: Leonid
"""

import numpy as np
from gp_list import *
from gp_tree import *
import random
import re

#====================================================================================================================
def one_point_recombination(parent1, parent2):
    #одноточечное скрещивание.Выбираем общие дуги из общих дуг случайно выбираем
    #точку разрыва. Выбираем основного родителя, делаем его копию. Заменяем 
    #соответствующее поддерево от второго родителя
    
    parent1_noms=parent1.get_tree_number()
    parent2_noms=parent2.get_tree_number()
    
    parent1_noms=re.sub('(\)|\(|\,|\s)', '_', parent1_noms)
    parent1_noms=re.split('_+', parent1_noms)
    if '' in parent1_noms:
        parent1_noms.remove('')
    
    parent2_noms=re.sub('(\)|\(|\,|\s)', '_', parent2_noms)
    parent2_noms=re.split('_+', parent2_noms)
    if '' in parent2_noms:
        parent2_noms.remove('')
    
    
    general_noms = set(parent1_noms).intersection(parent2_noms)
    general_noms=list(general_noms)
    # print('общие дуги: \n',general_noms)
    if len(general_noms)<2:
        raise RuntimeError("Ошибка рекомбинации деревье {0} \n {1}".format(parent1.print_tree(), parent2.print_tree()))
        return False
    
    
    random_index = np.random.randint(len(general_noms))
    # print('выбрали индекс: \n',random_index)
    nom=general_noms[random_index]
    # print('выбрали номер: \n',nom)
    
    if np.random.rand()<0.5:
        node=parent1.get_node(nom)
        node1=node.copy()
        parent_rez=parent2.copy()
        parent_rez.recombination(node1, nom)
    else:
        node=parent2.get_node(nom)
        node1=node.copy()
        parent_rez=parent1.copy()
        parent_rez.recombination(node1, nom)
        
    
    return parent_rez

#====================================================================================================================
def standart_recombination(parent1, parent2):
    #стандартное скрещивание.Выбираем случайные дуги у обоих родителей в качестве
    #точек разрыва. Выбираем основного родителя, делаем его копию. Заменяем 
    #соответствующее поддерево от второго родителя
    
    parent1_noms=parent1.get_tree_number()
    parent2_noms=parent2.get_tree_number()
    
    parent1_noms=rez=re.sub('(\)|\(|\,|\s)', '_', parent1_noms)
    parent1_noms=re.split('_+', parent1_noms)
    if '' in parent1_noms:
        parent1_noms.remove('')
    else:
        mus=10
    parent2_noms=rez=re.sub('(\)|\(|\,|\s)', '_', parent2_noms)
    parent2_noms=re.split('_+', parent2_noms)
    if '' in parent2_noms:
        parent2_noms.remove('')
    else:
        mus=10
    
    
    if len(parent1_noms)<1:
        raise RuntimeError("Ошибка рекомбинации деревье {0} \n {1}".format(parent1.print_tree(), parent2.print_tree()))
        return False
    
    if len(parent2_noms)<1:
        raise RuntimeError("Ошибка рекомбинации деревье {0} \n {1}".format(parent1.print_tree(), parent2.print_tree()))
        return False
    
    #индекс 0 не генерируем, т.к. будет пустая строка
    random_index = np.random.randint(len(parent1_noms))
    # print('Родитель1 : \n',parent1_noms)
    # print('выбрали индекс: \n',random_index)
    nom1=parent1_noms[random_index]
    # print('выбрали номер: \n',nom1)
    
    #индекс 0 не генерируем, т.к. будет пустая строка
    random_index = np.random.randint(len(parent2_noms))
    # print('Родитель2 : \n',parent2_noms)
    # print('выбрали индекс: \n',random_index)
    nom2=parent2_noms[random_index]
    # print('выбрали номер: \n',nom2)
    
    if np.random.rand()<0.5:
        node=parent1.get_node(nom1)
        node1=node.copy()
        parent_rez=parent2.copy()
        parent_rez.recombination(node1, nom2)
    else:
        node=parent2.get_node(nom2)
        node1=node.copy()
        parent_rez=parent1.copy()
        parent_rez.recombination(node1, nom1)
    return parent_rez

def recombination(parent1, parent2, type_recombination='one_point'):
    if type_recombination=='one_point':
        rez=one_point_recombination(parent1, parent2)
    elif type_recombination=='standart':
        rez=standart_recombination(parent1, parent2)
    else:
        raise RuntimeError("неизвестный тип рекомбинации: {0}".format(type_recombination))
        return False
    return rez

if __name__=='__main__':
    list_F=[]
    list_F.append(list_sum(num_childs=1))
    list_F.append(list_sum(num_childs=2))
    list_F.append(list_sum(num_childs=3))
    
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
    parent1=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
                  limit_level=20)
    parent2=gp_tree(list_T=list_T, list_F=list_F, level=0, nom_list='1', type_ini='full',
                  limit_level=20)
    
    print(parent1.print_tree())
    print(parent2.print_tree())
    #==========================================================================
    parent_rez=one_point_recombination(parent1, parent2)
    # parent_rez=standart_recombination(parent1, parent2)
    print(parent_rez.print_tree())
    