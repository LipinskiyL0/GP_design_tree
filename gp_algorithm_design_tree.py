'''
В данном файле описан класс реализующий генетическое программирование для поиска эффективных 
деревьев решения, которое наследуется от базового класса ГП gp_algorithm
Основное отличие в вычислении функции пригодности
'''


from gp_algorithm import gp_algorithm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from gp_tree_design_tree import gp_tree_design_tree
from gp_list_design_tree import *
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import KFold

class gp_algorithm_design_tree (gp_algorithm):
    def fit_function(self, tree, params):
        #вычисление пригодности индивида

        X=params['X']
        y=params['y']
        n_split=params['n_split']

        mas_loss=np.zeros(n_split, dtype=np.float32)
        kf = KFold(n_splits=n_split, shuffle=True)

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            # print(f"Fold {i}:")
            e=tree.fit(X=X.iloc[ train_index, :], y=y.iloc[train_index], method=params['method'],
                    metric=params['score_metric'], iterations=params['iterations'])
            mas_loss[i]=tree.loss(x0=None, X=X.iloc[ test_index, :], y=y.iloc[test_index], 
                                  metric=params['score_metric'], list_keys=None)
            
        fit=1/(1+np.mean(mas_loss))
        num_node=tree.get_num_node()
        if ('penalty_num_node' in params) == False:
            k_penalty=0
        else:
            k_penalty=params['penalty_num_node']
        fit=fit-k_penalty*num_node
        if fit<0:
            fit=0
        return fit
if __name__ == '__main__':
    data = load_diabetes()
    X=data.data
    X=pd.DataFrame(X, columns=data.feature_names)
    y=data.target
    y=pd.Series(y, name='y')
    # params={'X':X, 'y':y,'method':'DE','score_metric':'f1','iterations':20   }
    params={'X':X, 'y':y,'mask':None,'epsilon':1e-10, 'num_samples':4,'inf_name':'mse', 'list_F':None, 'list_T':None,
            'n_features':1,'method':'self_optimization','score_metric':'mse','iterations':50, 'n_split':5,
              'penalty_num_node':0.00000001  }
    list_T=[]
    list_T.append(list_regr_const())

    list_F=[]
    for c in X.columns:
        list_F.append(list_less(name_feature=c ))
    
    gp=gp_algorithm_design_tree(gp_tree_design_tree, n_ind=20, n_iter=20,
                                 list_T=list_T, list_F=list_F, type_ini='LearnID3',
                                   limit_level=4, params=params)
    
    rez=gp.opt()
    print('лучшая пригодность: ', rez['fit'])
    print('лучшее решение: ', rez['individ'].print_tree())
    e=rez['individ'].score(X=X, y=y, metric='f1')
    print(f'точность работы на исходном дереве: {e}')
