# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:30:16 2023

в этом файле реализуются селекции различного вида для эволюционных процедур
общая формула селекции: на входе массив с пригодностями, на выходе индексы пар родителей

@author: Leonid
"""
import numpy as np
from scipy.stats import rankdata

#==============================================================================
def selectionTour(Fit, n_tur=5):
    #турнирная селекция
    if len(Fit)<2:
        raise RuntimeError("пустой массив в селекцию: {0}".format(Fit))
        return False
    #создаем родительские пары
    ind_per=np.zeros([len(Fit), 2], dtype=int)
    for i in range(len(Fit)):
        for j in range(2):
            tour=np.random.randint(0, len(Fit), n_tur)
            ftour=Fit[tour]
            index=np.argmax(ftour)
            ind_per[i, j]=tour[index]
        # if i%100==0:
        #     print(i)
    return ind_per
#==============================================================================
def selectionProp(Fit):
    #пропорциональная селекция
    if len(Fit)<2:
        raise RuntimeError("пустой массив в селекцию: {0}".format(Fit))
        return False
    
    cum_Fit=np.cumsum(Fit)
    indexes=np.arange(0, len(Fit), 1, dtype=int)
    summa=sum(Fit)
    #создаем родительские пары
    ind_per=np.zeros([len(Fit), 2], dtype=int)
    for i in range(len(Fit)):
        for j in range(2):
            s=np.random.rand()*summa
            ind=cum_Fit<s
            index=indexes[ind]
            if s<=cum_Fit[0]:
                index=0
            else:
                index=index[-1]
            ind_per[i, j]=index
        # if i%100==0:
        #     print(i)
    return ind_per

#==============================================================================
def selectionRang(Fit):
    #Ранговая селекция
    #использует пропорциональную селекцию
    if len(Fit)<2:
        raise RuntimeError("пустой массив в селекцию: {0}".format(Fit))
        return False
    Fit_rank=rankdata(Fit)
    rez=selectionProp(Fit_rank)
    
    return rez
#==============================================================================
def selection(Fit, type_selection='tournament', n_tur=5):
    
    if type_selection=='tournament':
        rez=selectionTour(Fit, n_tur)
    elif type_selection=='proportional':
        rez=selectionProp(Fit)
    elif type_selection=='ranking':
        rez=selectionRang(Fit)
    else:
        raise RuntimeError("неизвестный тип селекции: {0}".format(type_selection))
        return False
    return rez

if __name__=='__main__':
    
    Fit=np.random.rand(10000)
    # rez=selectionTour(Fit, 5)
    # rez=selectionProp(Fit)
    rez=selectionRang(Fit)
    
    # for i in range(len(rez)):
    #     print('{0}, {1}, {2}, {3}'.format(rez[i,0], rez[i,1], Fit[rez[i,0]], Fit[rez[i,1]]))
    