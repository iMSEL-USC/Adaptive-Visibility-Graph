#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:09:03 2022

@author: junlinou
"""


import numpy as np
import math
from numba import cuda
from numba import jit
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32



@cuda.jit#('(float32[:, :, :], float32[:, :], int32[:], float32[:, :, :], float32[:, :])')
def population_path_free_G(new_population_out, obstacles_out, num_edge_out, rng_states, pop_out):
    x, y = cuda.grid(2)
    # Thread id in a 2D block
    blockId = (cuda.gridDim.x * cuda.blockIdx.y) + cuda.blockIdx.x
    threadId = (blockId * (cuda.blockDim.x * cuda.blockDim.y)) + (cuda.threadIdx.y * cuda.blockDim.x) + cuda.threadIdx.x
    
    for i in range(2, new_population_out.shape[2] - 2, 2):
        c = 1
        while c%2 == 1:
            n2 = 0
            pop_x = (pop_out[x][i] + xoroshiro128p_normal_float32(rng_states, threadId))%100
            pop_y = (pop_out[x][i+1]+ xoroshiro128p_normal_float32(rng_states, threadId))%100


            for m in range(num_edge_out.shape[0]):
                c = 0
                for n in range(num_edge_out[m]):
                    n1 = (n+1)%num_edge_out[m]
                    A_x = obstacles_out[n2+n, 0]
                    A_y = obstacles_out[n2+n, 1]
                    B_x = obstacles_out[n2+n1, 0]
                    B_y = obstacles_out[n2+n1, 1]

                    if (((pop_y > A_y) != (pop_y > B_y)) and ((pop_x - A_x)*(B_y - A_y)*(B_y - A_y) < (B_x - A_x) * (pop_y - A_y)*(B_y - A_y))):
                        c += 1
                if c%2 == 1:
                    break
                n2 += num_edge_out[m]
        if y == 0:
            new_population_out[x,y,i] = pop_out[x,i]
            new_population_out[x,y,i + 1] = pop_out[x,i+1]
        else:
            new_population_out[x,y,i] = pop_x
            new_population_out[x,y,i + 1] = pop_y



@cuda.jit('(float32[:, :, :], float32[:, :], int32[:], int32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :])')
def fitness(pop, obstacles_out, num_edge_out, intersection_value_out, length_out, smoothness_out, safety_out, fitness_value_out):
    # 2 D coordinates
    x, y = cuda.grid(2)
    #lamda = 1.5
    summ = 0
    length = 0
    smoothness = 0
    safety = 0
    lamda = 1
    #length_out[x,y] = 0
    #safety_out[x,y] = 0
    #smoothness_out[x,y] = 0
    intersection_value_out[x, y] = 1
    dx_1 = pop[x,y,2]-pop[x,y,0]
    dy_1 = pop[x,y,3]-pop[x,y,1]

    for z in range(int(pop.shape[2]/2) - 1):
        if z < int(pop.shape[2]/2) - 2:
            dx_2 = pop[x,y,2*z+4]-pop[x,y,2*z+2]
            dy_2 = pop[x,y,2*z+5]-pop[x,y,2*z+3]
            direction = 180 * abs(math.atan2(dy_2*dx_1-dx_2*dy_1,dx_1*dx_2+dy_1*dy_2))/math.pi
            if dx_2!=0 or dy_2!=0:
                dx_1 = dx_2
                dy_1 = dy_2
            smoothness += direction
        A_x = pop[x,y,2*z]*10
        A_y = pop[x,y,2*z+1]*10
        B_x = pop[x,y,2*z+2]*10
        B_y = pop[x,y,2*z+3]*10
        AB = math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))/10
        #min_value = 10000
        n = 0
        
        for i in range(num_edge_out.shape[0]):
            for j in range(num_edge_out[i]):
                n1 = (j+1)%num_edge_out[i]
            
                C_x = obstacles_out[n+j, 0]*10
                C_y = obstacles_out[n+j, 1]*10
                D_x = obstacles_out[n+n1, 0]*10
                D_y = obstacles_out[n+n1, 1]*10
                
                j1 = (B_x-A_x)*(C_y-A_y)-(B_y-A_y)*(C_x-A_x)
                j2 = (B_x-A_x)*(D_y-A_y)-(B_y-A_y)*(D_x-A_x)
                j3 = (D_x-C_x)*(A_y-C_y)-(D_y-C_y)*(A_x-C_x)
                j4 = (D_x-C_x)*(B_y-C_y)-(D_y-C_y)*(B_x-C_x)

                    
                if j1*j2<=0 and j3*j4<=0 and (B_x-A_x)**2+(B_y-A_y)**2!=0:
                    summ += 1
            n += num_edge_out[i]
        length += AB
    if summ == 0:
        intersection_value_out[x, y] = 0
    # if the path does not intersect with the obstacles, we need to calculate the shortest distance between segments of path and obtacles
    if intersection_value_out[x, y] == 0:
        for z in range(int(pop.shape[2]/2) - 1):
            # set a large value for sub
            sub = np.inf
            # set initial value for n
            n = 0
            #coodinates of points A, B
            A_x = pop[x,y,2*z]
            A_y = pop[x,y,2*z+1]
            B_x = pop[x,y,2*z+2]
            B_y = pop[x,y,2*z+3]
            # length of AB
            AB = math.sqrt((B_x-A_x)**2 + (B_y-A_y)**2)
            for i in range(num_edge_out.shape[0]):
                for j in range(num_edge_out[i]):
                    n1 = (j+1)%num_edge_out[i]
                    #dist_out[x][y][i][j][k] = 0
                    #coodinates of points C, D
                    C_x = obstacles_out[n+j, 0]
                    C_y = obstacles_out[n+j, 1]
                    D_x = obstacles_out[n+n1, 0]
                    D_y = obstacles_out[n+n1, 1]
                    #length of CD,......
                    CD = math.sqrt((C_x - D_x)**2 + (C_y - D_y)**2)
                    AC = math.sqrt((A_x - C_x)**2 + (A_y - C_y)**2)
                    AD = math.sqrt((A_x - D_x)**2 + (A_y - D_y)**2)
                    BC = math.sqrt((B_x - C_x)**2 + (B_y - C_y)**2)
                    BD = math.sqrt((B_x - D_x)**2 + (B_y - D_y)**2)
                    temp1 = np.inf
                    temp2 = np.inf
                    if AB != 0:
                        #condition 1
                        r_1 = ((C_x - A_x) * (B_x - A_x) + (C_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_1 <= 0:
                            temp1 = AC
                        elif r_1 >= 1:
                            temp1 = BC
                        else:
                            tem = r_1 * AB
                            temp1 = math.sqrt(max(0,AC * AC - tem * tem))
                        #condition 2
                        r_2 = ((D_x - A_x) * (B_x - A_x) + (D_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_2 <= 0:
                            temp2 = AD
                        elif r_2 >= 1:
                            temp2 = BD
                        else:
                            tem = r_2 * AB
                            temp2 = math.sqrt(max(0,AD * AD - tem * tem))    
                    #condition 3
                    r_3 = ((A_x - C_x) * (D_x - C_x) + (A_y - C_y) * (D_y - C_y))/(CD * CD)
                    if r_3 <= 0:
                        temp3 = AC
                    elif r_3 >= 1:
                        temp3 = AD
                    else:
                        tem = r_3 * CD
                        temp3 = math.sqrt(max(0,AC * AC - tem * tem))
                    #condition 4
                    r_4 = ((B_x - C_x) * (D_x - C_x) + (B_y - C_y) * (D_y - C_y))/(CD * CD)
                    if r_4 <= 0:
                        temp4 = BC
                    elif r_4 >= 1:
                        temp4 = BD
                    else:
                        tem = r_4 * CD
                        temp4 = math.sqrt(max(0,BC * BC - tem * tem))
                    # minimum distance between two segments
                    temp = min(temp1, temp2, temp3, temp4)

                    # obtain the shortest the distance
                    if sub > temp:
                        sub = temp
                # for the next obstacle
                n += num_edge_out[i]
                # calculate the safety value for the obstacles
            if sub<lamda:
                safety += math.exp(lamda-sub)
    smoothness_out[x,y] = smoothness
    length_out[x,y] = length
    safety_out[x,y] = safety
    fitness_value_out[x][y] = 0.1 * length + 0.1 * smoothness  + 0.8 * safety + 10000 * intersection_value_out[x, y]

@cuda.jit('(int32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :])')
def fitness_calculation(intersection_value_out, length_out, safety_out, smoothness_out, fitness_value_out):
    x, y = cuda.grid(2)
    
    fitness_value_out[x][y] = 0.5 * length_out[x][y] + 0.1 * smoothness_out[x][y] + 0.4 * safety_out[x][y] + 1000 * intersection_value_out[x][y]
    
@cuda.jit('(float32[:, :, :], float32[:, :], float32[:, :], float32[:, :, :])')
def selection(new_population_out, fitness_value_out, fitness_out, parents_out):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    x, y = cuda.grid(2)
    summ = 0

    for i in range(fitness_value_out.shape[1]):
        if fitness_value_out[x][i] < fitness_value_out[x][y] or fitness_value_out[x][i] == fitness_value_out[x][y] and i < y:
            summ += 1
    fitness_out[x][summ] = fitness_value_out[x][y]
    for j in range(new_population_out.shape[2]):
        parents_out[x][summ][j] = new_population_out[x][y][j]
        
     

@cuda.jit('(float32[:, :], int32, float32[:], int32[:])')
def selection2(fitness_out, generation, trend_out,order):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    x = cuda.grid(1)
    summ = 0

    for i in range(fitness_out.shape[0]):
        if fitness_out[i][0] < fitness_out[x][0] or fitness_out[i][0] == fitness_out[x][0] and i < x:
            summ += 1
    if summ == 0:
        trend_out[generation] = fitness_out[x][0]
        order[0] = x
            

@cuda.jit('(float32[:, :, :], float32[:, :, :])')
def crossover(parents_out, offspring_out):

    # The point at which crossover takes place between two parents. Usually it is at the center.
    x, y = cuda.grid(2)
    num_offspring = offspring_out.shape[1]
    crossover_point = int(parents_out.shape[2]/4)
    # Index of the first parent to mate.
    parent1_idx = 2 * y%num_offspring
    # Index of the second parent to mate.
    parent2_idx = (2 * y + 1)%num_offspring
    for i in range(crossover_point):
        # The first new offspring will have its first half of its genes taken from the first parent.
        offspring_out[x][2 * y][2 * i] = parents_out[x][parent1_idx][2 * i]
        offspring_out[x][2 * y][2 * i + 1] = parents_out[x][parent1_idx][2 * i + 1]
        # The first offspring will have its second half of its genes taken from the second parent.
        offspring_out[x][2 * y][2 * (crossover_point + i)] = parents_out[x][parent2_idx][2 * (crossover_point + i)]
        offspring_out[x][2 * y][2 * (crossover_point + i) + 1] = parents_out[x][parent2_idx][2 * (crossover_point + i) + 1]
        # The second offspring will have its first half of its genes taken from the first parent.
        offspring_out[x][2 * y + 1][2 * i] = parents_out[x][parent2_idx][2 * i]
        offspring_out[x][2 * y + 1][2 * i + 1] = parents_out[x][parent2_idx][2 * i + 1]
        # The second offspring will have its second half of its genes taken from the second parent.
        offspring_out[x][2 * y + 1][2 * (crossover_point + i)] = parents_out[x][parent1_idx][2 * (crossover_point + i)]
        offspring_out[x][2 * y + 1][2 * (crossover_point + i) + 1] = parents_out[x][parent1_idx][2 * (crossover_point + i) + 1]
    
    offspring_out[x][2 * y][parents_out.shape[2] - 2] = parents_out[x][parent2_idx][parents_out.shape[2] - 2]
    offspring_out[x][2 * y][parents_out.shape[2] - 1] = parents_out[x][parent2_idx][parents_out.shape[2] - 1]
    offspring_out[x][2 * y + 1][parents_out.shape[2] - 2] = parents_out[x][parent1_idx][parents_out.shape[2] - 2]
    offspring_out[x][2 * y + 1][parents_out.shape[2] - 1] = parents_out[x][parent1_idx][parents_out.shape[2] - 1]



@cuda.jit('(float32[:, :, :],  float32[:, :, :], float32[:, :, :])')
def new_popul(parents_out, offspring_out, new_population_out):
    x, y = cuda.grid(2)
    num_offspring = offspring_out.shape[1]
    if y<num_offspring * 0.5:
        for i in range(parents_out.shape[2]):
            new_population_out[x][y][i] = parents_out[x][y][i]
    else:
        for i in range(parents_out.shape[2]):
            new_population_out[x][y][i] = offspring_out[x][y-int(num_offspring * 0.5)][i]



@cuda.jit#('(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :], int32[:], int32)')
def mutation_free(rng_states, new_population_out, obstacles_out, num_edge_out, generation):
    # Mutation changes a single gene in each offspring randomly.
    idx, idy = cuda.grid(2)
    # Thread id in a 2D block
    blockId = (cuda.gridDim.x * cuda.blockIdx.y) + cuda.blockIdx.x
    threadId = (blockId * (cuda.blockDim.x * cuda.blockDim.y)) + (cuda.threadIdx.y * cuda.blockDim.x) + cuda.threadIdx.x
    
    c = 1
    if idy >= int(0.1 * new_population_out.shape[1]):
        for m in range(2, new_population_out.shape[2]-2):
            if xoroshiro128p_uniform_float32(rng_states, threadId)<0.06:
                while c%2 == 1:
                    bx = m
                    if bx%2 == 0:
                        y = bx + 1
                        x = bx
                        nx = (new_population_out[idx][idy][x] + xoroshiro128p_normal_float32(rng_states, threadId))%100
                        ny = new_population_out[idx][idy][y]
                    else:
                        y = bx
                        x = y - 1
                        nx = new_population_out[idx][idy][x]
                        ny = (new_population_out[idx][idy][y] + xoroshiro128p_normal_float32(rng_states, threadId))%100
                
                    n = 0
                    for i in range(num_edge_out.shape[0]):
                        c = 0
                        for j in range(num_edge_out[i]):
                            n1 = (j+1)%num_edge_out[i]
                            
                            A_x = obstacles_out[n+j, 0]
                            A_y = obstacles_out[n+j, 1]
                            B_x = obstacles_out[n+n1, 0]
                            B_y = obstacles_out[n+n1, 1]

                            if (((ny > A_y) != (ny > B_y)) and ((nx - A_x)*(B_y - A_y)*(B_y - A_y) < (B_x - A_x) * (ny - A_y)*(B_y - A_y))):
                                c += 1

                        if c%2 == 1:
                            break
                        n += num_edge_out[i]
                new_population_out[idx][idy][x] = nx
                new_population_out[idx][idy][y] = ny



@cuda.jit('(float32[:, :, :], float32[:, :, :], int32)')
def migration(new_population_out, parents_out, generation):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    x,y = cuda.grid(2)
    # Index of the next population
    if (generation+1)%10 == 0 and y < int(0.1 * new_population_out.shape[1]):
        population_idx = (x + 1)%parents_out.shape[0]
        for i in range(new_population_out.shape[2]):
            new_population_out[x][y][i] = parents_out[population_idx][y][i]

