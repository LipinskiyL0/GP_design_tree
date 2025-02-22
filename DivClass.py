# -*- coding: utf-8 -*-
"""
В классе реализуется алгоритм дифференциальной эволюции

"""


import numpy as np 
import matplotlib.pyplot as plt

def FitFunct(ind, one, two):
    return np.sum(ind**4)

class DivClass:
    def __init__(self):#конструктор класса
        self.f=0.5   #Параметр создания мутантного вектора   
        self.p=0.5   #вероятность скрещивания
        self.n_ind=100   #количество индивидов
        self.n_iter=10    #количество итераций
        self.Min=0       #минимум при стартовой генерации
        self.Max=1       #максимум при стартовой генерации
        self.FitFunk=0   #функция ошибки
    
#==============================================================================        
    def inicialization(self, FitFunct, Min, Max, n_ind=100, args=()):
        #инициализация популяции
        
        self.n_ind=n_ind   #количество индивидов
        self.Min=np.array(Min)       #минимум при стартовой генерации
        self.Max=np.array(Max)      #максимум при стартовой генерации
        self.FitFunct=FitFunct   #функция ошибки
        
        
        if np.sum(np.array(self.Min>=self.Max, dtype=int))>0:
            return False
        
        
         #стартовая генерация
        self.pop=np.random.sample((n_ind, len(self.Min)))
        #привеодим в заданный диапазон
        Mean=self.Max-self.Min
        self.pop=self.pop*Mean[np.newaxis, :]-self.Min[np.newaxis, :]        
        # self.fit=np.array(list(map(self.FitFunct, self.pop)))
        self.fit=[]
        for ii in range(len(self.pop)):
            self.fit.append(self.FitFunct(self.pop[ii, :], *args))
        self.fit=np.array(self.fit)

        # print("start evolution")
        # print("iteration 0: min={0}, mean={1}, max={2}".format(np.min(self.fit),np.mean(self.fit),
        #                                 np.max(self.fit)))        
        
        return True
 #============================================================================   
    def opt(self, n_iter=10, f=0.2, p=0.25, args=()):
        #оптимизация
        self.f=f   #Параметр создания мутантного вектора   
        self.p=p   #вероятность скрещивания
        self.n_iter=n_iter    #количество итераций
        
        for i in range(self.n_iter-1):
            
            #создаем мутантный вектор
            index=np.random.randint(self.n_ind, size=(self.n_ind, 3))
            mut_pop=self.pop[index[:, 0], :]+0.5*(self.pop[index[:, 1], :]-self.pop[index[:, 2], :])
            
            #скрещивание
            p=np.random.sample((self.n_ind, len(self.Min)))
            mask=p<self.p
            mut_pop[mask]=self.pop[mask]
            
            #производим оценку пробных векторов
            fit_new=[]
            for ii in range(len(mut_pop)):
                fit_new.append(self.FitFunct(mut_pop[ii, :], *args))
            fit_new=np.array(fit_new)
            # fit_new=np.array(list(map(self.FitFunct, mut_pop)))
            
            #находим пробные вектора которые лучше, чем соответствующие исходные
            mask=fit_new<self.fit
            self.pop[mask, :]=mut_pop[mask, :]
            self.fit[mask]=fit_new[mask]
            # print("iteration {3}: min={0}, mean={1}, max={2}".format(np.min(self.fit),np.mean(self.fit),
            #                             np.max(self.fit), i+1)) 
            ind=np.argmin(self.fit)

        return self.pop[ind, :]

if __name__ == '__main__':        
    Div=DivClass()
    Div.inicialization(FitFunct=FitFunct, Min=[-100, -100], Max=[100, 100], args=('one', 'two'))  
    rez=Div.opt(n_iter=20, args=('one', 'two'))     
    print(rez)
    
        
        
        
        























