# import numpy as np
# from matplotlib import pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# x_bound=[-2048,2048]
# y_bound=[-2048,2048]
# dna_size=24
# pop_size=100
# epoch=100
# def F(x,y):
#     return 100*(x**2-y**2)+(1-x)**2
# def plot_3d(ax):
#     x=np.linspace(*x_bound,100)
#     y=np.linspace(*y_bound,100)
#     x,y=np.meshgrid(x,y)
#     z=F(x,y)
#     ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='coolwarm')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title('mzy')
#     plt.pause(0.1)
#     plt.show()
# def jiema(pop):
#     x_pop=pop[:,1::2]
#     y_pop=pop[:,0::2]
#     x_pop=np.dot(x_pop, 2 ** np.arange(dna_size)[::-1])
#     x_pop=x_pop/(2**dna_size-1)
#     x_pop=x_pop*(x_bound[1]-x_bound[0])+x_bound[0]
#     y_pop = np.dot(y_pop, 2 ** np.arange(dna_size)[::-1])
#     y_pop = y_pop / (2 ** dna_size - 1)
#     y_pop = y_pop * (y_bound[1] - y_bound[0]) + y_bound[0]
#     return x_pop,y_pop
# def jiaochabianyi(pop):
#     newpop=[]
#     for father in pop:
#         child=father
#         if np.random.rand()<0.7:
#             mother=pop[np.random.randint(pop_size)]
#             point=np.random.randint(0,dna_size*2)
#             child[point:]=mother[point:]
#         if np.random.rand()<0.1:
#             point2=np.random.randint(0,dna_size*2)
#             child[point2]=child[point2]^1
#         newpop.append(child)
#     return np.array(newpop)
# def select(pop):
#     x,y=jiema(pop)
#     fit =(F(x, y) - min(F(x, y)) + 1e-3)
#     fitrate=fit/sum(fit)
#     index=np.random.choice(pop_size, pop_size, True, fitrate)
#     return pop[index]
#
#
#
# if __name__== '__main__':
#     # fig = plt.figure()
#     ax=plt.subplot(projection='3d')
#     plt.ion()
#     plot_3d(ax)
#     pop=np.random.randint(0,2,(pop_size,dna_size*2))
#     for i in range(epoch):
#       x,y=jiema(pop)
#       # if 'sca' in locals():
#       #     sca.remove()
#       sca=ax.scatter(x,y,F(x,y),'b*')
#       plt.pause(0.1)
#       plt.show()
#       sca.remove()
#       pop=jiaochabianyi(pop)
#       pop=select(pop)
#     fit = (F(x, y) - min(F(x, y))) + 0.001
#     max_fitness_index=np.argmax(fit)
#     print("max_fitness:", fit[max_fitness_index])
#     print("最优的基因型：", pop[max_fitness_index])
#     print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
#     plt.ioff()
#     sca = ax.scatter(x, y, F(x, y), 'b*')
#     plot_3d(ax)
'''接下来是
    复现文章部分
'''
import time

import numpy as np
import matplotlib.pyplot as plt
starttime=time.time()
lung={0.167:0.681,0.5:0.436,1:0.709,2:0.263,6:0.12}
DNA_SIZE = 20
POP_SIZE = 150
CROSSOVER_RATE = 0.95
MUTATION_RATE = 0.00001
N_GENERATIONS = 1000
P1_BOUND = [0, 1]
P2_BOUND = [0, 1]
Q1_BOUND = [0, 1]
Q1_BOUND = [0, 1]
def F(p1,p2,q1,q2):
    z=0
    for i,j in lung.items():
         z+=(p1*np.exp(-q1*i)+p2*np.exp(-q2*i)-j)**2
    return z
def jiema(pop):
    p1_pop=pop[:,::4]
    p2_pop = pop[:, 1::4]
    q1_pop = pop[:, 2::4]
    q2_pop = pop[:, 3::4]
    p1_pop=p1_pop.dot(2**np.arange(DNA_SIZE)[::-1])/(2**DNA_SIZE-1)
    p2_pop=p2_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2 ** DNA_SIZE - 1)
    q1_pop=q1_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2 ** DNA_SIZE - 1)
    q2_pop=q2_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2 ** DNA_SIZE - 1)
    return p1_pop,p2_pop,q1_pop,q2_pop
def fitness(pop):
    p1,p2,q1,q2=jiema(pop)
    return max(F(p1,p2,q1,q2))-F(p1,p2,q1,q2)+1e-3
def jiaochabianyi(pop):
    new_pop=[]
    MUTATION_RATE=0.00001
    for father in pop:
        child=father
        if np.random.rand()<CROSSOVER_RATE:
            mother=pop[np.random.randint(0,POP_SIZE)]
            point=np.random.randint(0,DNA_SIZE*4)
            child[point:]=mother[point:]
        child=mutation1(child,MUTATION_RATE)
        MUTATION_RATE=MUTATION_RATE*2
        new_pop.append(child)
    return np.array(new_pop)
def mutation1(child,MUTATION_RATE):
    if np.random.rand() < MUTATION_RATE:
        point = np.random.randint(0, DNA_SIZE * 4)
        child[point] = child[point] ^ 1
    return child

def mutation2(child, MUTATION_RATE):
    mutate_point = np.random.randint(0, DNA_SIZE * 4)  # 随机产生变异基因的位置
    if mutate_point >= DNA_SIZE / 2:
        MUTATION_RATE = MUTATION_RATE * 2
    if np.random.rand() < MUTATION_RATE:
        child[mutate_point] = child[mutate_point] ^ 1
    return child
def select(pop):
    fit=fitness(pop)
    idx=np.random.choice(range(POP_SIZE),POP_SIZE,replace=True,p=fit/sum(fit))
    return pop[idx]
def info(pop):
    fit=fitness(pop)
    a=np.argmax(fit)
    print(max(fit))
    p1,p2,p3,p4=jiema(pop)
    print(p1[a],p2[a],p3[a],p4[a])
    print(F(p1[a],p2[a],p3[a],p4[a]))

pop=np.random.randint(0,2,(POP_SIZE,DNA_SIZE*4))
for i in range(N_GENERATIONS):
    pop=jiaochabianyi(pop)
    select(pop)
info(pop)
endtime=time.time()
print('运行时间为',endtime-starttime)
