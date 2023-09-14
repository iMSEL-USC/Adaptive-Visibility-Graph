#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:13:14 2022

@author: junlinou
"""

from numba import cuda
import math

@cuda.jit('(float32[:, :], float32[:, :], int32[:], int32[:, :], float32[:, :])')
def intersection(points_out, obstacles_out, num_edge_out, intersection_value_out, dist_out):
    # 2 D coordinates
    x, y = cuda.grid(2)
    if x < points_out.shape[0] and y < points_out.shape[0]:
        summ = 0
        n = 0
        
        intersection_value_out[x, y] = 1
        A_x = points_out[x, 0]*10
        A_y = points_out[x, 1]*10
        B_x = points_out[y, 0]*10
        B_y = points_out[y, 1]*10
        distAB = math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))/10
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
    
                if j1*j2<=0 and j3*j4<=0:
                    summ += 1
                
            n += num_edge_out[i]
        if summ == 0 and x!= y:
            intersection_value_out[x, y] = 0
            dist_out[x,y] = distAB
