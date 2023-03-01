# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:17:56 2023

@author: g
"""
# 复现https://blog.csdn.net/baidu/article/details/124432689?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167767253516800211530675%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167767253516800211530675&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-124432689-null-null.142^v73^control,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95%E8%A7%A3%E5%86%B3tsp%20python&spm=1018.2226.3001.4187
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sko.GA import GA_TSP

# 输入数据
# %%有起始点运行以下代码
num_points = 6  # 点的数量
points_coordinate = np.random.rand(num_points, 2)  # 随机生成点的位置
start_point=[[0,0]]  # 起始点
end_point=[[1,1]]  # 末尾点
points_coordinate=np.concatenate([points_coordinate,start_point,end_point])  # 所有点进行组合
# %%无起始点运行此代码
points_coordinate = np.array([[1.304, 2.231], [3.639, 1.315], [4.177, 2.244], [3.712, 1.399], [3.488, 1.535], [3.326, 1.556],
                            [3.238,1.229],[4.196,1.044],[4.312,0.79],[4.386,0.57],[3.007,1.97],[2.562,1.756],[2.788,1.491],
                            [2.381,1.676],[1.332,0.695],[3.715,1.678],[3.918,2.179],[4.061,2.37],[3.78,2.212],[3.676,2.578],
                            [4.029,2.838],[4.263,2.931],[3.429,1.908],[3.507,2.376],[3.394,2.643],[3.439,3.201],[2.935,3.24],
                            [3.14,3.55],[2.545,2.357],[2.778,2.826],[2.37,2.975]])
num_points = len(points_coordinate)
# %%
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points,  = routine.shape
    # start_point,end_point 本身不参与优化。给一个固定的值，参与计算总路径
    routine=np.concatenate([routine,routine[0,np.newaxis]])
    return sum([distance_matrix[routine[i], routine[i + 1]] for i in range(num_points+1-1)])#


ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=100, max_iter=2000, prob_mut=0.01)
best_points, best_distance = ga_tsp.run()


fig, ax = plt.subplots(1, 2)
best_points=np.concatenate([best_points,best_points[0,np.newaxis]])
best_points_coordinate = points_coordinate[best_points, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()