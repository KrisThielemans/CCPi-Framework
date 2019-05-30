#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:30:57 2019

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
import pywt

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
AcquisitionData, DataContainer, BlockGeometry
from ccpi.optimisation.operators import Gradient, BlockOperator
from ccpi.astra.operators import AstraProjectorSimple
from ccpi.optimisation.operators import FiniteDiff
from ccpi.optimisation.algorithms import SBTV


# create phantom
N = 75
phantom = np.zeros((N,N))
phantom[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
phantom[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

'''
# show Ground Truth
plt.imshow(phantom)
plt.title('Ground Truth')
plt.show()
'''

data = ImageData(phantom)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = N
angles = np.linspace(0, np.pi, round(N / 4), dtype = np.float32)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)
Aop = AstraProjectorSimple(ig, ag, 'cpu')

# generate sinogram
g = Aop.direct(data)
'''
# show Sinogram
plt.imshow(g.as_array())
plt.title('Sinogram')
plt.colorbar()
plt.show()
'''

# scale coefficient - used to scale Aop.adjoint(Aop.direct(x))
scale = (np.pi / 2) / np.sqrt(N * N)

# init
x1 = Aop.adjoint(g) * scale
'''
# show Initialization image
plt.imshow(x.as_array())
plt.title('Initialization image')
plt.colorbar()
plt.show()
'''

# initialize some variables
BG = BlockGeometry(ig, ig)
s = BG.allocate(value = 0)

div_tol = 1e-12

# constants
# Split Bregman maximu number of iterations
sb_iter = 100
# Split-Bregman mu
sb_mu = 10
# Split Bregman lambda
sb_lambda = 50
# Split Bregman tolerance
sb_tol = 1e-5 * x1.norm()
# Gradient Descent alpha (rate)
gd_alpha = 1e-3
# Gradient Descent number of iterations
gd_iter = 30

# initialize some variables
x = x1.copy()
k = 0
err = x.norm()

# initialize Wavelet operator
frame = 'db1'
level = 2
wavelet = pywt.Wavelet(frame)
print(wavelet)
coeffs = pywt.wavedec(x1.as_array(), wavelet, level = level)


s = []
# initialize s
for i in range(len(coeffs)):
    s.append(np.zeros(coeffs[i].shape, dtype = float))
    
d = []
# initialize s
for i in range(len(coeffs)):
    d.append(np.zeros(coeffs[i].shape, dtype = float))


while ((err > sb_tol) and (k < sb_iter)):
    
    x_0 = x.copy()
    
    wtx = pywt.wavedec(x.as_array(), wavelet, level = level)
    
    # update d
    for i in range(len(coeffs)):
        d[i] = np.sign(wtx[i] + s[i]) * np.maximum(np.abs(wtx[i] + s[i]) - 1 / sb_mu, 0)

    # update x using Gradient Descent
    for i in range(gd_iter):
        tmp = []
        for j in range(len(coeffs)):
            tmp.append(wtx[j] - d[j] + s[j])

        x -= gd_alpha * (sb_lambda * scale * Aop.adjoint(Aop.direct(x) - g) + 
                         sb_mu * ImageData(array = pywt.waverec(tmp, wavelet)[:N, :N]))

    err = (x - x_0).norm()
    k += 1
    
    print('iter {}, err {}'.format(k, err))
    
    # update s
    wtx = pywt.wavedec(x.as_array(), wavelet, level = level)

    for i in range(len(coeffs)):
        s[i] += wtx[i] - d[i]


# Show Ground Truth and Reconstruction
plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(x.as_array())
plt.title('Reconstruction')
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
plt.plot(np.linspace(0, N, N), data.as_array()[int(N/2), :], label = 'Ground Truth')
plt.plot(np.linspace(0, N, N), x.as_array()[int(N/2), :], label = 'Reconstruction')
plt.legend()
plt.title('Horizontal Line Profiles')
plt.subplot(2,1,2)
plt.plot(np.linspace(0, N, N), data.as_array()[:, int(N/2)], label = 'Ground Truth')
plt.plot(np.linspace(0, N, N), x.as_array()[:, int(N/2)], label = 'Reconstruction')
plt.legend()
plt.title('Verical Line Profiles')
plt.show()
