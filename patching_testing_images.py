#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:56:43 2023

@author: anaworker
"""

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff

large_image_stack = tiff.imread('testing.tif') #165 slika dimenzija 768x1024 (HxW)
large_mask_stack = tiff.imread('testing_groundtruth.tif')

# =============================================================================
# large_image_stack.shape[0] #165 slika
# 
# large_image = large_image_stack[0]
# large_image_stack[0].shape
# patches_img = patchify(large_image, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
# patches_img.shape[0]
# 
# patches_img.shape[1]
# =============================================================================


#      ------            ------           ------


for img in range(large_image_stack.shape[0]):

    large_image = large_image_stack[img]
    
    patches_img = patchify(large_image, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite('patches/test_images2/' + 'timage_' + str(img) + '_' + str(i)+str(j)+ ".tif", single_patch_img)
