# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:15:28 2023

@author: g
"""

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sko.GA import GA_TSP


num_points = 20  # 点的数量

points_coordinate = np.random.rand(num_points, 2)  # 随机生成点的位置
start_point=[[0,0]]  # 起始点
end_point=[[1,1]]  # 末尾点
points_coordinate=np.concatenate([points_coordinate,start_point,end_point])  # 所有点进行组合
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')  # 计算距离矩阵


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points,  = routine.shape
    # start_point,end_point 本身不参与优化。给一个固定的值，参与计算总路径
    routine = np.concatenate([[num_points], routine, [num_points+1]])
    return sum([distance_matrix[routine[i], routine[i + 1]] for i in range(num_points+2-1)])


ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=0.01)
best_points, best_distance = ga_tsp.run()


fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([[num_points], best_points, [num_points+1]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()