import cv2
import numpy as np
import mahotas
from skimage.feature import local_binary_pattern

class Descriptor:
    
    @classmethod
    def _loadImage(cls,imgPath):
        img = cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (100, 100)) 
#         norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return img
        
    @classmethod
    def getHOGDescriptor(cls,filename):
        img = cls._loadImage(filename)
        descriptor = cls._getHOGVector(img)
        return descriptor
    
    @classmethod
    def _getHOGVector(cls,image):
        winSize = (100,100)
        blockSize = (20,20)
        blockStride = (10,10)
        cellSize = (10,10)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    
        descriptor = hog.compute(image)
        desc= []
        for x in range(len(descriptor)):
            desc.append(descriptor[x,0])
        return desc
    
    @classmethod
    def getLocalBinaryPatterns(cls,filename):
        img = cls._loadImage(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        points = 16
        radius =2
#         hist = mahotas.features.lbp(gray, radius, points)

        feat = local_binary_pattern(gray, points, radius)
        (hist, aaaa) = np.histogram(feat.ravel(),
            bins=np.arange(0, points + 3),
            range=(0, points + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum())
        return hist

#     @classmethod
#     def getHaralick(cls,filename):
#         img = cls._loadImage(filename)
#         # convert the image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # compute the haralick texture feature vector
#         haralick = mahotas.features.haralick(gray).mean(axis=0)
#         return haralick

#     @classmethod
#     def getHistogram(cls,filename):
#         img = cls._loadImage(filename)
#         color = ('b','g','r')
#         for i,col in enumerate(color):
#             hist = cv2.calcHist([img],[i],None,[256],[0,256])
#         data = np.array(hist)
#         descriptor = data.flatten()
#         return descriptor

#     def getHuMoments(self,filename):
#         img = self._loadImage(filename)
#         image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         feature = cv2.HuMoments(cv2.moments(image)).flatten()
#         return feature
#     

#     
#     def getMixShapePatter(self,filename):
#         haralick = self.getHaralick(filename)
#         hu = self.getHuMoments(filename)
#         desc =  np.concatenate((haralick,hu), axis=0)
#         return desc
        
    

    
    