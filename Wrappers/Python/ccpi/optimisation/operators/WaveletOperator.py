#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:07:42 2019

@author: evelina
"""

from ccpi.optimisation.operators import Operator, LinearOperator, ScaledOperator
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry
import numpy as np
import pywt


class WaveletOperator(LinearOperator):
    CORRELATION_SPACE = "Space"
    CORRELATION_SPACECHANNEL = "SpaceChannels"

    def __init__(self, gm_domain, **kwargs):
        
        self.gm_domain = gm_domain
        self.correlation = kwargs.get('correlation', WaveletOperator.CORRELATION_SPACE)
        self.level = kwargs.get('level', 1)
        self.frame = kwargs.get('frame', 'haar')
        
        if self.correlation == WaveletOperator.CORRELATION_SPACE:
            if self.gm_domain.channels > 1:
                if self.gm_domain.length == 4:
                    # 3D + Channel
                    # expected order = ['channels', 'direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    order = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                else:
                    # 2D + Channel
                    # expected order = ['channels', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    order = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                self.ind = order[1:]
            else:
                # no channel info
                if self.gm_domain.length == 3:
                    # 3D
                    # expected order = ['direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    order = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                else:
                    # 2D
                    expected_order = [ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    order = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                self.ind = order
        elif self.correlation == WaveletOperator.CORRELATION_SPACECHANNEL:
            if self.gm_domain.channels > 1:
                if self.gm_domain.length == 4:
                    # 3D + Channel
                    # expected order = ['channels', 'direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    order = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                else:
                    # 2D + Channel
                    # expected order = ['channels', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    order = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                self.ind = order
            else:
                raise ValueError('No channels to correlate')
        
        # Call WaveletTransform operator
        self.wavelet = pywt.Wavelet(self.frame)
        arr, self.coeff_slices = pywt.coeffs_to_array(pywt.wavedecn(self.gm_domain.allocate(value = 0).as_array(), 
                                                                    self.wavelet, 
                                                                    axes = self.ind, 
                                                                    level = self.level), axes = self.ind)
        if self.gm_domain.channels > 1:
            if self.gm_domain.length == 4:
                self.gm_range =  ImageGeometry(channels = arr.shape[order[0]],
                                               voxel_num_z = arr.shape[order[1]],
                                               voxel_num_y = arr.shape[order[2]],
                                               voxel_num_x = arr.shape[order[3]])
            else:
                self.gm_range =  ImageGeometry(channels = arr.shape[order[0]],
                                               voxel_num_y = arr.shape[order[1]],
                                               voxel_num_x = arr.shape[order[2]])
        else:
            if self.gm_domain.length == 3:
                self.gm_range =  ImageGeometry(voxel_num_z = arr.shape[order[0]],
                                               voxel_num_y = arr.shape[order[1]],
                                               voxel_num_x = arr.shape[order[2]])
            else:
                self.gm_range =  ImageGeometry(voxel_num_y = arr.shape[order[0]],
                                               voxel_num_x = arr.shape[order[1]])
    
    def direct(self, 
               x, 
               out = None):
                
        if out is not None:
            tmp, coeff_slices = pywt.coeffs_to_array(pywt.wavedecn(x.as_array(), self.wavelet, level = self.level, axes = self.ind), 
                                                     axes = self.ind)
            out.array = tmp
        else:
            tmp, coeff_slices = pywt.coeffs_to_array(pywt.wavedecn(x.as_array(), self.wavelet, level = self.level, axes = self.ind), 
                                                     axes = self.ind)
            return ImageData(array = tmp,
                             geometry = self.gm_range)
    
    def adjoint(self, 
                x, 
                out = None):
        
        if out is not None:
            tmp = pywt.waverecn(pywt.array_to_coeffs(x.as_array(), coeff_slices = self.coeff_slices), 
                                self.wavelet, 
                                axes = self.ind)
            if self.gm_domain.length == 4:
                out.array = tmp[:self.gm_domain.shape[0],\
                                :self.gm_domain.shape[1],\
                                :self.gm_domain.shape[2],\
                                :self.gm_domain.shape[3]]
            elif self.gm_domain.length == 3:
                out.array = tmp[:self.gm_domain.shape[0],\
                                :self.gm_domain.shape[1],\
                                :self.gm_domain.shape[2]]
            else:
                out.array = tmp[:self.gm_domain.shape[0],\
                                :self.gm_domain.shape[1]]
        else:
            tmp = pywt.waverecn(pywt.array_to_coeffs(x.as_array(), coeff_slices = self.coeff_slices), 
                                self.wavelet, 
                                axes = self.ind)
            if self.gm_domain.length == 4:
                return ImageData(array = tmp[:self.gm_domain.shape[0],\
                                             :self.gm_domain.shape[1],\
                                             :self.gm_domain.shape[2],\
                                             :self.gm_domain.shape[3]],
                                 geometry = self.gm_domain,
                                 dimension_labels = self.gm_domain.dimension_labels)
            elif self.gm_domain.length == 3:
                return ImageData(array = tmp[:self.gm_domain.shape[0],\
                                             :self.gm_domain.shape[1],\
                                             :self.gm_domain.shape[2]],
                                 geometry = self.gm_domain,
                                 dimension_labels = self.gm_domain.dimension_labels)
            else:
                return ImageData(array = tmp[:self.gm_domain.shape[0],\
                                             :self.gm_domain.shape[1]],
                                 geometry = self.gm_domain,
                                 dimension_labels = self.gm_domain.dimension_labels)
    
    def domain_geometry(self):
        return self.gm_domain

    def range_geometry(self):
        return self.gm_range
    
    def PowerMethod(operator, iterations, x_init=None):
        raise NotImplementedError
        
    def PowerMethodNonsquare(op,numiters , x_init=None):
        raise NotImplementedError

