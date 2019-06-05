import cv2
import os
import re
import numpy as np
import random
import mahotas


class Data:
    def __init__(self, location = "data"):
        self.path = os.path.join(os.path.dirname(__file__), location)
        self.uniqueLabels=None
        self.trainingSet=None
        self.validationSet=None
        self.testingSet=None
        
        
    def preprocessing(self,functionDescriptor1):

        filenameTraining, labelTraining, filenameValidate, labelValidate =  self._getLearningImages()
        filenameTesting, labelTesting = self._getTestingImages()
        
        #load training data
        self.trainingSet = self._getImageData(filenameTraining, labelTraining, functionDescriptor1)
        
        #load validate data
        self.validationSet= self._getImageData(filenameValidate, labelValidate, functionDescriptor1)
        
        #load test data
        self.testingSet = self._getImageData(filenameTesting, labelTesting, functionDescriptor1)
        print("Loading Data Completed")

    ## return array of [x,y] which are numpy arrays
    def getTrainingSet(self):
        return [np.vstack(self.trainingSet[0]),np.vstack(self.trainingSet[1])]
    def getValidationSet(self):
        return [np.vstack(self.validationSet[0]),np.vstack(self.validationSet[1])]
    def getTestingSet(self):
        return [np.vstack(self.testingSet[0]),np.vstack(self.testingSet[1])]
    def getUniqueLabels(self):
        return self.uniqueLabels
    
    def _loadImage(self,imgPath):
        img = cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (100, 100)) 
        return img
    
    #return tables of images' paths and their labels
    def _getLearningImages(self):
        imgPath = os.path.join(self.path, "learn")
        classes = sorted(os.walk(imgPath).__next__()[1])
        self.uniqueLabels = classes
    
        trainingSetX= []
        trainingSetY= []
        validationSetX = []
        validationSetY = []
        for c in classes:
                imagesPath, imagesLabel = list(), list()
                c_dir = os.path.join(imgPath, c)
                walk = os.walk(c_dir).__next__()
                label = ([0])*len(classes)
                label[classes.index(c)]=1.0
                for sample in walk[2]:
                    if sample.endswith('.jpg') or sample.endswith('.jpeg') or sample.endswith('.png'):
                        imagesPath.append(os.path.join(c_dir, sample))
                        imagesLabel.append(label)
                        
                Lx, Ly, Vx,Vy = self._splitData(imagesPath, imagesLabel)
                trainingSetX.extend(Lx)
                trainingSetY.extend(Ly)
                validationSetX.extend(Vx)
                validationSetY.extend(Vy)         
        return trainingSetX, trainingSetY, validationSetX, validationSetY
    
    def _getTestingImages(self):
        imgPath = os.path.join(self.path, "test")
        classes = sorted(os.walk(imgPath).__next__()[1])
        if(classes != self.uniqueLabels):
            print("test and learn classes are miscellaneous!")
        classes = sorted(os.walk(imgPath).__next__()[1])
        imagesPath, imagesLabel = list(), list()
        for c in classes:   
                c_dir = os.path.join(imgPath, c)
                walk = os.walk(c_dir).__next__()
                label = ([0])*len(classes)
                label[classes.index(c)]=1.0
                for sample in walk[2]:
                    if sample.endswith('.jpg') or sample.endswith('.jpeg') or sample.endswith('.png'):
                        imagesPath.append(os.path.join(c_dir, sample))
                        imagesLabel.append(label)                         
        return imagesPath, imagesLabel
        
    #return shuffled numpy table of 2 lists X,Y, of features and labels
    def _getImageData(self,filenames, labels, functionDescriptor):
        features= []
        for c in filenames:
            features.append(functionDescriptor(c))
            
        features,label = self._shuffleData(features,labels)
        dataset = np.array([features,label])
        return dataset
    
    def _shuffleData(self, tableX, tableY):
        data = list(zip(tableX,tableY))
        random.shuffle(data)
        x, y = list(zip(*data))
        return x,y
    
        #slit 80% data for learning and 20 for testing
    def _splitData(self,images, labels):
        x, y = self._shuffleData(images, labels)   
        learnX = x[0:int(0.8*len(images))]
        learnY = y[0:int(0.8*len(images))]
        valX = x[int(0.8*len(images)):]
        valY = y[int(0.8*len(images)):]
        return learnX, learnY, valX, valY


############################
def loadImage(imgPath):
    img = cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (100, 100)) 
    return img    
 

    
#############################################################
###################     DESCRIPTORS   #######################
def getHOGDescriptor(filename):
    img = loadImage(filename)
    descriptor = getHOGVector(img)
    return descriptor


def getPixel(filename):
    img = loadImage(filename)
    img = cv2.resize(img, (20, 20))
    data = np.array(img)
    descriptor = data.flatten()
    
    return descriptor


def getHistogram(filename):
    img = loadImage(filename)
    color = ('b','g','r')
    for i,col in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
#     print(hist.shape)
    data = np.array(hist)
    descriptor = data.flatten()
    return descriptor

def getHuMoments(filename):
    img = loadImage(filename)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def getHaralick(filename):
    img = loadImage(filename)
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def getMixShapePatter(filename):
    haralick = getHaralick(filename)
    hu = getHuMoments(filename)
    desc =  np.concatenate((haralick,hu), axis=0)
    return desc
    

def getLocalBinaryPatterns(filename):
    img = loadImage(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = 8
    radius =1
    hist = mahotas.features.lbp(gray, radius, points)
    return hist

def getHOGVector(image):
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
    
def createLabels(imageNames):
    unique = set(imageNames)
    outputVector = ([len(unique)])*len(imageNames)
    #return output array to nerual network from names
    for elem in range(len(imageNames)):
        vec=[]
        for i in unique:
            if imageNames[elem] == i:
                vec.append(1.0)
            else:
                vec.append(0.0)
        outputVector[elem]=vec
    return outputVector
    
def getClassName(name):
    result = name.rsplit( ".", 1 )[ 0 ]
    result = re.sub("[^a-zA-Z]+", "", result)
    return result

# def getDescriptors(img):
#     orb = cv2.ORB_create()
#     # returns descriptors of an image
#     return orb.detectAndCompute(img, None)[1]