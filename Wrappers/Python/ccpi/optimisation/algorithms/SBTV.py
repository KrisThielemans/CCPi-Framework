#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:17:36 2019

@author: evelina
"""

from ccpi.optimisation.algorithms import Algorithm
import numpy as np

from ccpi.optimisation.operators import Gradient
from ccpi.framework import DataContainer, BlockGeometry, AcquisitionData


class SB(Algorithm):
    
    def __init__(self, **kwargs):
        '''initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        super(SB, self).__init__()
        
        self.g = kwargs.get('g', None)
        self.operator = kwargs.get('operator', None)
        self.x_init = kwargs.get('x_init', None)
        self.sb_mu = kwargs.get('sb_mu', 1)
        self.sb_lambda = kwargs.get('sb_lambda', 1)
        self.sb_tol = kwargs.get('sb_tol', None)
        self.gd_rate = kwargs.get('gd_rate', 1e-3)
        self.gd_iter = kwargs.get('gd_iter', 100)
        
        if self.g is not None and self.operator is not None:
            print ("Calling from creator")
            self.set_up(self.g,
                        self.operator,
                        self.x_init,
                        self.sb_mu, 
                        self.sb_lambda,
                        self.sb_tol,
                        self.gd_rate,
                        self.gd_iter)

    def set_up(self, 
               g, 
               operator, 
               x_init = None, 
               sb_mu = 1, 
               sb_lambda = 1, 
               sb_tol = None,
               gd_rate = 1e-3,
               gd_iter = 100):
        
        # algorithmic parameters
        self.g = g
        self.operator = operator
        self.sb_mu = sb_mu
        self.sb_lambda = sb_lambda
        self.sb_tol = sb_tol
        self.gd_rate = gd_rate
        self.gd_iter = gd_iter
        
        if x_init is None:
            self.x_init = self.operator.domain_geometry().allocate(value = 0)
            
        else:
            self.x_init = x_init.copy()
        
        self._ig = self.operator.domain_geometry()
        self.x = self.x_init.copy()
        self.x_0 = self.x_init.copy()
        
        if isinstance(g, AcquisitionData):
            # TODO: was tested only with parallel geometry
            self._scale = (np.pi / 2) / g.geometry.pixel_num_h
        else:
            self._scale = 1
            
        # initialize Gradient operator
        self._gradient = Gradient(self._ig, correlation='Space')
        
        # initialize auxiliary variables
        BG = BlockGeometry(self._ig, self._ig)
        self._s = BG.allocate(value = 0)
        self._d = self._ig.allocate(value = 0)
        self._err = self.x.norm()

    def update(self):
        self._x_0 = self.x.copy()
        
        # update d 
        grad_x = self._gradient.direct(self.x)
        h = (grad_x + self._s).pnorm(2)
        self._d = (h - 1 / self.sb_mu).maximum(0) * (grad_x + self._s) / h.maximum(1e-12)
        
        # update x using Gradient Descent
        for i in range(self.gd_iter):
            self.x -= self.gd_rate * (self.sb_lambda * self._scale * self.operator.adjoint(self.operator.direct(self.x) - self.g) + 
                                      self.sb_mu * (self._gradient.adjoint(self._gradient.direct(self.x) - self._d + self._s)))
    
        # update s
        grad_x = self._gradient.direct(self.x)
        self._s += grad_x - self._d
        
        self._err = (self.x - self._x_0).norm()
    
    def update_objective(self):
        self.loss.append(self._err)
        
    def should_stop(self):
        '''stopping cryterion, currently only based on number of iterations'''
        if (self.sb_tol is not None):
            return self._err <= self.sb_tol
    
    