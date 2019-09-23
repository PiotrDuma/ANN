import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class Descriptor:
    
    @classmethod
    def _loadImage(cls,imgPath):
        try:
            img = cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (100, 100)) 
        except Exception as e:
            print("Error with: ", imgPath)
            print(str(e))
            
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
    def LBP(cls,filename):
        img = cls._loadImage(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        radius =2
        points = 16*radius

#         hist = mahotas.features.lbp(gray, radius, points)

        feat = local_binary_pattern(gray, points, radius)
        (hist, aaaa) = np.histogram(feat.ravel(),
            bins=np.arange(0, points + 3),
            range=(0, points + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum())
        return hist


    @staticmethod
    def getLocalBinaryPatterns(filename):
        img = Descriptor._loadImage(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        radius =2
        points = 8
        
        dsc = local_binary_pattern(img, points, radius)        

        (hist, aaaa) = np.histogram(dsc.ravel(),
        bins=np.arange(0, 2**points),
        range=(0, dsc.max()+1))
    
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum())
        return hist
  
    @classmethod
    def getHistogram(cls,filename):
        img = cls._loadImage(filename)
        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
        data = np.array(hist)
        descriptor = data.flatten()
        descriptor = descriptor.astype("float")
        descriptor /= (descriptor.sum())
        return descriptor

    @classmethod
    def mix(cls,filename):
        hist = cls.getHistogram(filename)
        hog = cls.getHOGDescriptor(filename)
        lbp = cls.getLocalBinaryPatterns(filename)

        descriptor = np.concatenate((hist,lbp,hog))
        return descriptor




    
    
    