#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:47:17 2023

@author: ouj
"""

import numpy as np
from numba import njit
import math
import random

# Checking if values are in an array
#from the website: https://stackoverflow.com/questions/54930852/python-numba-value-in-array
@njit
def isin(val, arr):
    for i in range(len(arr)):
        if arr[i] == val:
            return True
    return False

#from statistics import median
@njit
def minCost(dist,h,queue):
    # Initialize min value and min_index as -1
    minimum = math.inf
    min_index = -1
          
    # from the dist array,pick one which
    # has min value and is till in queue
    for i in range(len(dist)):
        if dist[i]+h[i] < minimum and isin(i,queue):
            minimum = dist[i]+h[i]
            min_index = i
    return min_index

# A* algorithm
@njit
def astar(graph, h, src, tar):
  
    row = graph.shape[0]
    col = row
    path = np.ones(0).astype(np.int64)
    # The output array. dist[i] will hold
    # the shortest distance from src to i
    # Initialize all distances as INFINITE 
    dist = math.inf * np.ones(row)
  
    #Parent array to store 
    # shortest path tree
    parent = -1 * np.ones(row).astype(np.int64)
  
    # Distance of source vertex 
    # from itself is always 0
    dist[src] = 0
    # Add all vertices in queue
    queue = np.arange(row).astype(np.int64)
    
    u = -1
    #Find shortest path for all vertices
    while u != tar:
  
        # Pick the minimum dist vertex
        # from the set of vertices
        # still in queue
        u = minCost(dist,h,queue)
        #print(u)
        # remove min element
        if u == tar:
            break
        if u < 0:
            tar = 0
            break
        queue = queue[queue != u]
  
        # Update dist value and parent 
        # index of the adjacent vertices of
        # the picked vertex. Consider only 
        # those vertices which are still in
        # queue
        for i in range(col):
            '''Update dist[i] only if it is in queue, there is
            an edge from u to i, and total weight of path from
            src to i through u is smaller than current value of
            dist[i]'''
            if graph[u][i]!=0 and isin(i,queue):
                if dist[u] + graph[u][i] < dist[i]:
                    dist[i] = dist[u] + graph[u][i]
                    parent[i] = u
    j = tar
    pa = path
    while j != src:
        path = np.append(pa,j)
        j = parent[j]
        pa = path
    path = np.append(path,src)
    #path = getPath(path, parent, tar)
    return path



@njit
def med_fitness_paths(dist, h, src, tar, num, points):
    path_feasible = []
    num_p = np.zeros(num).astype(np.int64)
    fitness_p = np.zeros(num).astype(np.float32)
    p = 0
    a = 0
    i = 0
    while(i<num):
        #if i == 1:
        #    p = 0.8
        dist1 = dist.copy()
        for m in range(len(dist)):
            for n in range(m,len(dist)):
                if random.random() < p:
                    dist1[m,n] = 0
                    dist1[n,m] = 0
        pa = astar(dist1,h,src,tar)
        if pa[0] == 0 and a <= 0:
            return np.zeros((1,1)).astype(np.float32), fitness_p
        if pa[0] == 0:
            a -= 0.1
            p = a**3
        else:
            a += 0.1
            p = a**3
            path_feasible.append(pa)
            num_p[i] = len(pa)
            fitness_p[i] = fitness(pa,points)
            i += 1
    fitness_sort = np.sort(fitness_p)
    for i in range(len(path_feasible)-1,-1,-1):
        if fitness_p[i]>fitness_sort[int(num/8-1)]:
            path_feasible.pop(i)
            num_p = np.delete(num_p, i)
            fitness_p = np.delete(fitness_p, i)

    num_point = max(num_p)
    path = np.zeros((len(path_feasible),int(2*num_point))).astype(np.float32)
    for i in range(len(path_feasible)):
        r = np.array([random.randint(0, len(path_feasible[i])-2) for k in range(int(num_point-len(path_feasible[i])))])
        #r = np.random.randint(len(path_feasible[i])-1, size=int(num_median-len(path_feasible[i])))
        s =  np.sort(r)
        k = 0
        n = 0
        for j in range(len(path_feasible[i])):
            path[i,2*n] = points[path_feasible[i][len(path_feasible[i])-j-1],0]
            path[i,2*n+1] = points[path_feasible[i][len(path_feasible[i])-j-1],1]
            n += 1
            if k < len(s):
                while(j==s[k]):
                    path[i,2*n] = (points[path_feasible[i][len(path_feasible[i])-j-1],0]+points[path_feasible[i][len(path_feasible[i])-j-2],0])/2
                    path[i,2*n+1] = (points[path_feasible[i][len(path_feasible[i])-j-1],1]+points[path_feasible[i][len(path_feasible[i])-j-2],1])/2
                    n += 1
                    k += 1
                    if k >= len(s):
                        break
    return path, fitness_p


@njit
def fitness(path,points):
    smoothness = 0
    length = 0
    dx_1 = points[path[1],0]-points[path[0],0]
    dy_1 = points[path[1],1]-points[path[0],1]
    for i in range(path.shape[0] - 1):
        if i < path.shape[0] - 2:
            dx_2 = points[path[i+2],0]-points[path[i+1],0]
            dy_2 = points[path[i+2],1]-points[path[i+1],1]
            direction = 180 * abs(math.atan2(dy_2*dx_1-dx_2*dy_1,dx_1*dx_2+dy_1*dy_2))/math.pi
            if (dx_2==0 and dy_2==0) == False:
                dx_1 = dx_2
                dy_1 = dy_2
            smoothness += direction
        A_x = points[path[i],0]
        A_y = points[path[i],1]
        B_x = points[path[i+1],0]
        B_y = points[path[i+1],1]
        length += math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))
    return (0.1*length + 0.8*smoothness)
