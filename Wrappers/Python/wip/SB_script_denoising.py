#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:51:09 2019

@author: evelina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:55:04 2019

@author: evelina
"""

import numpy as np 
import matplotlib.pyplot as plt
from skimage.util import random_noise

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
AcquisitionData, DataContainer, BlockGeometry
from ccpi.optimisation.operators import Gradient, Identity
from ccpi.astra.operators import AstraProjectorSimple
from ccpi.optimisation.operators import FiniteDiff

# create phantom
N = 75
phantom = np.zeros((N,N))
phantom[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
phantom[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

g = ImageData(random_noise(phantom, mode = 'gaussian', mean=0, var = 0.001, seed=10))

# show Ground Truth
plt.imshow(phantom)
plt.title('Ground Truth')
plt.show()

# show Noisy Image
plt.imshow(g.as_array())
plt.title('NoisyImage')
plt.show()

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Identity operator
Aop = Identity(ig)

# init
x = ig.allocate(value = 0)

# initialize some variables
BG = BlockGeometry(ig, ig)
s = BG.allocate(value = 0)

div_tol = 1e-12

# constants
# Split Bregman maximu number of iterations
sb_iter = 500
# Split-Bregman mu
sb_mu = 1
# Split Bregman lambda
sb_lambda = 5
# Split Bregman tolerance
sb_tol = 1e-5 * g.norm()
# Gradient Descent alpha (rate)
gd_alpha = 1e-2
# Gradient Descent number of iterations
gd_iter = 30

# initialize Gradient operator
gradient = Gradient(ig, correlation='Space')

# initialize other variables
k = 0
err = g.norm()
grad_x = gradient.direct(x)

while ((err > sb_tol) and (k < sb_iter)):
    
    x_0 = x.copy()
    
    # update d 
    h = grad_x + s
    d = h.sign()*(h.abs() - 1 / sb_mu).maximum(0)
    
    # update x using Gradient Descent
    for i in range(gd_iter):
        x -= gd_alpha * (sb_lambda * Aop.adjoint(Aop.direct(x) - g) + 
                         sb_mu * (gradient.adjoint(gradient.direct(x) - d + s)))
    
    err = (x - x_0).norm()
    k += 1
    
    print('iter {}, err {}'.format(k, err))
    
    # update s
    grad_x = gradient.direct(x)
    s += grad_x - d

# Show Ground Truth and Reconstruction
plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
plt.imshow(phantom)
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(x.as_array())
plt.title('Reconstruction')
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
plt.plot(np.linspace(0, N, N), phantom[int(N/2), :], label = 'Ground Truth')
plt.plot(np.linspace(0, N, N), x.as_array()[int(N/2), :], label = 'Reconstruction')
plt.legend()
plt.title('Horizontal Line Profiles')
plt.subplot(2,1,2)
plt.plot(np.linspace(0, N, N), phantom[:, int(N/2)], label = 'Ground Truth')
plt.plot(np.linspace(0, N, N), x.as_array()[:, int(N/2)], label = 'Reconstruction')
plt.legend()
plt.title('Verical Line Profiles')
plt.show()

'''
# %%%%%%%%%%%%%%%%%%%%%%%% old %%%%%%%%%%%%%%%%%%%%%%%%%%
# initialize some variables
sx = ig.allocate(value = 0)
sy = ig.allocate(value = 0)

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

# Show Ground Truth and Reconstruction
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(x.as_array())
plt.title('Reconstruction')
plt.colorbar()
plt.show()

plt.figure(2)
plt.subplot(1,2,1)
plt.plot(np.linspace(0, N, N), data.as_array()[int(N/2), :], label = 'Ground Truth')
plt.plot(np.linspace(0, N, N), x.as_array()[int(N/2), :], label = 'Reconstruction')
plt.legend()
plt.title('Horizontal Line Profiles')
plt.subplot(1,2,2)
plt.plot(np.linspace(0, N, N), data.as_array()[:, int(N/2)], label = 'Ground Truth')
plt.plot(np.linspace(0, N, N), x.as_array()[:, int(N/2)], label = 'Reconstruction')
plt.legend()
plt.title('Verical Line Profiles')
plt.show()
'''