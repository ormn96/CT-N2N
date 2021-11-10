# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:16:08 2021

@author: user
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def getPic(fileName):
    img = mpimg.imread(fileName)
    # img = img[:,:,1]
    # imgtoshow = img
    maxV = img.max()
    minV = img.min()
    return (img-minV)/(maxV-minV)

def showPic(img):
    plt.figure(dpi = 900)
    plt.imshow(img,cmap = 'gray')

def savePic(img,fileName):
    mpimg.imsave(fileName,img,cmap='gray',dpi=900)

def fixPic(fileName):
    img = mpimg.imread(fileName)
    mpimg.imsave("fix_"+fileName,img,cmap='gray',dpi=900)

# a = getPic('270.png')
# showPic(a)
fixPic('144.png')