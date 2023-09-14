#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:16:22 2023

@author: ouj
"""

#%%load module

import numpy as np
from numba import cuda
from timeit import default_timer as timer
import Extension
from matplotlib import pyplot as plt
import GA
import INTERSECTION as inters
import Astar
from numba.cuda.random import create_xoroshiro128p_states
#%%obstacles configuration

##fig. 6
obstacle = np.float32([[10,90],[10,50],[14,50],[14,80],[40,80],[40,90],
                        [50,60],[50,30],[52,30],[52,60],
                        [65,20],[65,10],[90,10],[90,50],[88,50],[88,20],
                        [50,70],[50,68],[52,68],[52,70]])
num_edge = np.int32([6,4,6,4]).astype(np.int32)
x_1 = 30
y_1 = 30
x_n = 80
y_n = 80

# basic setup
num_generations = 200
number_population = 64
number_candidate = 32
threads_per_block = (16, 16)#even number
blocks_per_grid = (int(number_population/threads_per_block[0]), int(number_candidate/threads_per_block[1]))

#generate population
start = timer()
# obstacle extension
L = -1.8
obstacles = Extension.exten(obstacle,num_edge,L)

start_point = np.float32([[x_1,y_1]])
goal_point = np.float32([[x_n,y_n]])
L = -1.801
obstacles_o = Extension.exten(obstacle,num_edge,L)
points = np.concatenate((np.concatenate((start_point,obstacles_o)),goal_point))

obstacles_out = cuda.to_device(obstacles)
points_out = cuda.to_device(points)
N = points.shape[0]
# get the visibility graph and h function for A*
intersect_value = np.ones((N,N)).astype(np.int32)
intersect_value_out = cuda.to_device(intersect_value)
dist = np.zeros((N,N)).astype(np.float32)
dist_out = cuda.to_device(dist)
h = np.zeros(N).astype(np.float32)
h_out = cuda.to_device(h)
num_edge_out = cuda.to_device(num_edge)
inters.intersection[(N,1),(1,N)](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out, h_out)
dist = dist_out.copy_to_host()
intersect_value = intersect_value_out.copy_to_host()
h = h_out.copy_to_host()

# number of paths needed to be obtained
num_path = 8 * number_population
# get the paths from A*
path, num_p = Astar.med_fitness_paths(dist, h, 0, N-1, num_path, points)
way_points = int(path.shape[1]/2)
number_of_genes = 2 * way_points


pop = path[0:number_population][:]
pop_out = cuda.to_device(pop)
#pop_new = GA_tenc.population_path_free(pop, obstacles, number_population, number_candidate, number_of_genes)
new_population = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
new_population[:, :, 0] = x_1
new_population[:, :, 1] = y_1
new_population[:, :, number_of_genes - 2] = x_n
new_population[:, :,number_of_genes - 1] = y_n
rng_states = create_xoroshiro128p_states(number_population * number_candidate, seed=1)

new_population_out = cuda.to_device(new_population)

GA.population_path_free_G[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, rng_states, pop_out)
cuda.synchronize()

#matrix initialization for genetic algorithm
intersection_value = np.ones((number_population, number_candidate)).astype(np.int32)
intersection_value_out = cuda.to_device(intersection_value)
#popul = np.empty((number, number_of_genes))
fitness = np.zeros((number_population, number_candidate)).astype(np.float32)
fitness_value_out = cuda.to_device(fitness)
fitness_out = cuda.device_array_like(fitness_value_out)
parents = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
parents_out = cuda.to_device(parents)
offspring_out = cuda.device_array_like(parents_out)


trend = np.zeros(num_generations+1).astype(np.float32)
trend_out = cuda.to_device(trend)
best_individual = np.zeros((number_of_genes)).astype(np.float32)
best_individual_out = cuda.to_device(best_individual)
order = np.zeros(1).astype(np.int32)
order_out = cuda.to_device(order)
            
length_out = cuda.device_array_like(fitness_value_out)
smoothness_out = cuda.device_array_like(fitness_value_out)
safety_out = cuda.device_array_like(fitness_value_out)


for generation in range(num_generations):
    # fitness calculation
    GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, intersection_value_out, length_out, smoothness_out, safety_out, fitness_value_out)
    #rank the individuals for each population
    GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
    #select the best individual
    GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
    #crossover
    GA.crossover[blocks_per_grid, (threads_per_block[0], int(threads_per_block[1]/2))](parents_out, offspring_out)
    #new population
    GA.new_popul[blocks_per_grid, threads_per_block](parents_out, offspring_out,new_population_out)
    #mutation
    GA.mutation_free[blocks_per_grid, threads_per_block](rng_states, new_population_out, obstacles_out, num_edge_out, generation)
    #migration
    GA.migration[blocks_per_grid, threads_per_block](new_population_out, parents_out, generation)
    #time_ga = timer() - start
    #print('Time taken for 10 is %f seconds.' % time_ga)
    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.


generation += 1
# fitness calculation
GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, intersection_value_out, length_out, smoothness_out, safety_out, fitness_value_out)
#rank the individuals for each population
GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
#select the best individual
GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
trend = trend_out.copy_to_host()
order = order_out.copy_to_host()
parents = parents_out.copy_to_host()
tem = order[0]
best_individual = parents[tem][0]
times = timer() - start
            

plt.figure("optimal path")
n = 0
for i in range(num_edge.shape[0]):
    plt.plot([obstacles[n+num_edge[i]-1,0],obstacles[n,0]], [obstacles[n+num_edge[i]-1,1],obstacles[n,1]],c="k")
    plt.plot(obstacles[n:n+num_edge[i],0], obstacles[n:n+num_edge[i],1],c="k")
    n += num_edge[i]

plt.plot(best_individual[range(0,len(best_individual),2)], best_individual[range(1,len(best_individual),2)],c="k")
plt.scatter(best_individual[range(0,len(best_individual),2)], best_individual[range(1,len(best_individual),2)],c="k")

plt.xlim(0, 100)
plt.ylim(0, 100)




