# GP_base
В этом репозитории реализован алгоритм GP на примере решения задачи деревьев решения
1. gp_list.py тут описываются вычислительные ядра для терминальных и функциональных узлов. 

2. gp_tree.py класс для описания навигации по дереву. Класс представляет собой узел дерева и ссылки на поддерево. В этом классе 
	берется вычислительная начинка из gp_list и оборачивается в функции вычисления и навигации по дереву. Дерево это набор классов gp_tree.
	Что бы "держать" дерево мы "держим" корневой узел, в котором содержатся потомки, в которых содержатся потомки и т.д. 

3. gp_algorithm класс для реализации эволюции над gp_tree
