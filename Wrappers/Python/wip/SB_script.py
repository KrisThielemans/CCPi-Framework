#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:55:04 2019

@author: evelina
"""

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData, DataContainer

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
phantom = np.zeros((N,N))
phantom[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
phantom[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

# show Ground Truth
plt.imshow(phantom)
plt.title('Ground Truth')
plt.show()

data = ImageData(phantom)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = N
angles = np.linspace(0, np.pi, N, dtype = np.float32)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)
Aop = AstraProjectorSimple(ig, ag, 'cpu')

# generate sinogram
b = Aop.direct(data)
# show Sinogram
plt.imshow(b.as_array())
plt.title('Sinogram')
plt.show()

# scale coefficient - used to scale Aop.adjoint(Aop.direct(x))
scale = 1 / np.sqrt(N * N)

# init
x = Aop.adjoint(b) * scale
# show Initialization image
plt.imshow(x.as_array())
plt.title('Initialization image')
plt.show()

# initialize some variables
sx = ImageData(array = np.zeros((N, N), dtype = np.float32))
sy = ImageData(array = np.zeros((N, N), dtype = np.float32))
div_tol = 1e-12

# constants
# number of iterations
sb_iter = 100
# Split-Bregman mu
sb_mu = 1
# Split Bregman lambda
sb_lambda = 1
# Split Bregman tolerance
sb_tol = 1e-5 * x.norm()
# Gradient Descent alpha
gd_alpha = 1e-3
# Gradient Descent number of iterations
gd_iter = 30

# initialize Finite Difference operators
FDx = FiniteDiff(ig, direction = 0, bnd_cond = 'Neumann')
FDy = FiniteDiff(ig, direction = 1, bnd_cond = 'Neumann')

# initialize other variables
k = 0
err = x.squared_norm()

while ((err > sb_tol) and (k < sb_iter)):
    
    x_0 = x.copy()
    
    # update d 
    gx = FDx.direct(x)
    gy = FDy.direct(x)
    h = ((gx + sx) ** 2 + (gy + sy) ** 2).sqrt()
    dx = (h - 1 / sb_mu).maximum(0) * (gx + sx) / h.maximum(div_tol)
    dy = (h - 1 / sb_mu).maximum(0) * (gy + sy) / h.maximum(div_tol)
    
    # update x using Gradient Descent
    for i in range(gd_iter):
        x -= gd_alpha * (sb_lambda * scale * Aop.adjoint(Aop.direct(x) - b) + 
                         sb_mu * (FDx.adjoint(FDx.direct(x) - dx + sx) + FDy.adjoint(FDy.direct(x) - dy + sy)))
    
    err = (x - x_0).squared_norm()
    k += 1
    
    print('iter {}, err {}'.format(k, err))
    
    # update s
    sx += FDx.direct(x) - dx
    sy += FDy.direct(x) - dy
    '''
    plt.imshow(x.as_array())
    plt.title('x iter {}'.format(k))
    plt.show()
    '''

# Show Ground Truth and Reconstruction
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(x.as_array())
plt.title('Reconstruction')
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'Ground Truth')
plt.plot(np.linspace(0,N,N), x.as_array()[int(N/2),:], label = 'Reconstruction')
plt.legend()
plt.title('Horizontal Line Profiles')
plt.subplot(2,1,2)
plt.plot(np.linspace(0,N,N), data.as_array()[:, int(N/2)], label = 'Ground Truth')
plt.plot(np.linspace(0,N,N), x.as_array()[:, int(N/2)], label = 'Reconstruction')
plt.legend()
plt.title('Verical Line Profiles')
plt.show()
