#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:17:36 2022

@author: junlinou
"""

import numpy as np
from numba import jit
@jit(nopython=True)
def exten(obstacles,num_edge, L):
#extension 
#   do extension of the boundary
    outer = np.zeros((obstacles.shape[0],2)).astype(np.float32)
    n = 0
    for i in range(num_edge.shape[0]):
        for j in range(num_edge[i]):
            n1 = (j-1)%num_edge[i]
            n2= (j+1)%num_edge[i]
            p1_x = obstacles[n+j, 0] - obstacles[n+n1, 0]
            p1_y = obstacles[n+j, 1] - obstacles[n+n1, 1]
            p2_x = obstacles[n+j, 0] - obstacles[n+n2, 0]
            p2_y = obstacles[n+j, 1] - obstacles[n+n2, 1]
            m1 = np.linalg.norm(np.array([p1_x,p1_y]),2)
            m2 = np.linalg.norm(np.array([p2_x,p2_y]),2)
            v1_x = p1_x/m1
            v1_y = p1_y/m1
            v2_x = p2_x/m2
            v2_y = p2_y/m2
            sintheta = np.cross(np.array([v1_x,-v1_y,0]),np.array([v2_x,-v2_y,0]))
            sin = sintheta[2]
            outer[n+j, 0] = -L/sin*(v1_x+v2_x) + obstacles[n+j, 0]
            outer[n+j, 1] = -L/sin*(v1_y+v2_y) + obstacles[n+j, 1]
        n += num_edge[i]
    return outer
    
