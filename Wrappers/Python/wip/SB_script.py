#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:55:04 2019

@author: evelina
"""

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np 
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import FISTA, CGLS, GradientDescent

from ccpi.optimisation.operators import Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, FunctionOperatorComposition
from skimage.util import random_noise
from ccpi.astra.operators import AstraProjectorSimple
from ccpi.optimisation.operators import FiniteDiff, SparseFiniteDiff

# create phantom
N = 75
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

data = ImageData(x)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = N
angles = np.linspace(0, np.pi, N, dtype = np.float32)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)
Aop = AstraProjectorSimple(ig, ag, 'cpu')

# generate sinogram
b = Aop.direct(data)

plt.imshow(b.as_array())
plt.title('sinogram')
plt.show()

scale = 1 / np.sqrt(N * N)

# init
x = Aop.adjoint(b) * scale

plt.imshow(x.as_array())
plt.title('init')
plt.show()

d_0 = np.zeros((N, N), dtype = np.float32)
# initialize some variables
sx = ImageData(array = np.zeros((N, N), dtype = np.float32))
sy = ImageData(array = np.zeros((N, N), dtype = np.float32))
zero_matrix = np.zeros((N, N), dtype = np.float32)
div_tol = 1e-12 * np.ones((N, N), dtype = np.float32)

# constants
# number of iterations
sb_iter = 100
# Split-Bregman mu
sb_mu = 1
# Split Bregman lambda
sb_lambda = 1
# Split Bregman tolerance
sb_tol = 1e-5 * np.sqrt(np.sum(np.sum(np.square(x), axis = 0), axis = 0))
# Gradient Descent alpha
gd_alpha = 1e-3
# Gradient Descent number of iterations
gd_iter = 30

# initialize Finite Difference operators
FDx = FiniteDiff(ig, direction = 0, bnd_cond = 'Neumann')
FDy = FiniteDiff(ig, direction = 1, bnd_cond = 'Neumann')

k = 0
err = np.sum(np.sum(np.square(x), axis = 0), axis = 0)

while ((err > sb_tol) and (k < sb_iter)):
    
    x_0 = x.copy()
    
    # update d 
    gx = FDx.direct(x)
    gy = FDy.direct(x)
    h = np.sqrt(np.square(gx.as_array() + sx.as_array()) + np.square(gy.as_array() + sy.as_array()))
    dx = ImageData(array = np.maximum(h - 1 / sb_mu, zero_matrix) * (gx.as_array() + sx.as_array()) / np.maximum(div_tol, h))
    dy = ImageData(array = np.maximum(h - 1 / sb_mu, zero_matrix) * (gy.as_array() + sy.as_array()) / np.maximum(div_tol, h))
    
    # update x using Gradient Descent
    for i in range(gd_iter):
        x -= gd_alpha * (sb_lambda * scale * Aop.adjoint(Aop.direct(x) - b) + sb_mu * (FDx.adjoint(FDx.direct(x) - dx + sx) + FDy.adjoint(FDy.direct(x) - dy + sy)))
    
    err = np.sum(np.sum(np.square(x.as_array() - x_0.as_array()), axis =0), axis =0)
    k += 1
    
    print('iter {}, err {}'.format(k, err))
    
    # update s
    sx = sx + FDx.direct(x) - dx
    sy = sy + FDy.direct(x) - dy
    '''
    plt.imshow(x.as_array())
    plt.title('x iter {}'.format(k))
    plt.show()
    '''

plt.imshow(x.as_array())
plt.title('res')
plt.show()
