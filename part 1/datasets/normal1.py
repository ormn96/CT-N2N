# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:16:08 2021

@author: user
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import random
import shutil
import cv2

def getPic2(fileName,outFileName):
    img = cv2.imread(fileName,-1)
    maxV = img.max()
    minV = img.min()
    imgOut = (img-minV)/(maxV-minV)
    cv2.imwrite(outFileName,imgOut)

    
def getPic(fileName,outFileName):
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

def fixPic(fileName,outFileName):
    img = mpimg.imread(fileName)
    mpimg.imsave(outFileName,img,cmap='gray',dpi=900)

# a = getPic('270.png')
# showPic(a)
#fixPic('144.png')


    
    
def crawl(inPath,outPath,fun):
    #os.mkdir(outPath)
    for root, dirs, files in os.walk(inPath):
        outPathFolder = root.replace(inPath, outPath)
        
        try:
            os.mkdir(outPathFolder)
        except FileExistsError:
            pass
        print(f"start folder: {root}")
        for f in files:
            outFilePath = os.path.join(outPathFolder,f)
            inFilePath = os.path.join(root,f)
            fun(inFilePath, outFilePath)
        print(f"end folder: {root}")
        
def sample(inPath,outPath,k):
        try:
                os.mkdir(outPath)
        except FileExistsError:
                pass
        for root, dirs, files in os.walk(inPath):
            print(f"start folder: {root}")
            if len(files)>k: 
                files_to_copy = random.sample(files, k)
                for f in files_to_copy:
                    s_path = os.path.join(root,f)
                    _, foldName = os.path.split(root)
                    o_path = os.path.join(outPath,f"{foldName}_{f}")
                    shutil.copyfile(s_path, o_path)
            print(f"end folder: {root}")


def shuffel(inPath):
    for root, dirs, files in os.walk(inPath):
        random.shuffle(files)
        for i,f in enumerate(files):
            s_path = os.path.join(root,f)
            o_path = os.path.join(root,f"{i}.png")
            shutil.move(s_path, o_path)
#crawl('./Images_png', './Images_png_out', fixPic)
#sample('./Images_png_out', './sample', 2)
#shuffel('./sample')

getPic2("../144.png","../1442.png")
img1 = cv2.imread("../1442.png",-1)
img2 = mpimg.imread("../fix_144.png")
diff = img1-img2[:,:,0]
print(diff)