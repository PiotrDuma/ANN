import cv2
import os
import re
import numpy as np
# from numpy.core.fromnumeric import size
import tensorflow as tf
# from numpy import shape
import random
# from cv2 import textureFlattening
import mahotas


def loadImage(imgPath):
    img = cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (100, 100))

    return img
    

def getImages(imgPath):
    classes = sorted(os.walk(imgPath).__next__()[1])
#     imagesPath, imagesLabel = list(), list()
    
#     print(classes)
#     for c in classes:
#             c_dir = os.path.join(imgPath, c)
#             walk = os.walk(c_dir).__next__()
#             label = ([0])*len(classes)
#             label[classes.index(c)]=1.0
#             for sample in walk[2]:
#                 if sample.endswith('.jpg') or sample.endswith('.jpeg'):
#                     imagesPath.append(os.path.join(c_dir, sample))
#                     imagesLabel.append(label)
#     return imagesPath, imagesLabel
        
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
                    
            Lx, Ly, Vx,Vy = splitData(imagesPath, imagesLabel)
            trainingSetX.extend(Lx)
            trainingSetY.extend(Ly)
            validationSetX.extend(Vx)
            validationSetY.extend(Vy)         
    return trainingSetX, trainingSetY, validationSetX, validationSetY

#slit 80% data for learning and 20 for testing
def splitData(images, labels):
    data = list(zip(images,labels))
    random.shuffle(data)

    x, y = list(zip(*data))
    
    learnX = x[0:int(0.8*len(images))]
    learnY = y[0:int(0.8*len(images))]
    
    valX = x[int(0.8*len(images)):]
    valY = y[int(0.8*len(images)):]
    return learnX, learnY, valX, valY

def preprocessing(imgPath, functionDescriptor1):
    filenameTraining, labelTraining, filenameValidate, labelValidate =  getImages(imgPath)

    #load training data
    trainingDataset = getData(filenameTraining, labelTraining, functionDescriptor1)
    
    #load test data
    validateDataset = getData(filenameValidate, labelValidate, functionDescriptor1)
     
    return trainingDataset,validateDataset

#descriptor as image for CNN
def getDescriptor(imgPath):
    filename, label = getImages(imgPath)
    dataset = tf.data.Dataset.from_tensor_slices((filename, label))
    
    def local(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        return image, label
    dataset = dataset.map(local)
    return dataset


def getData(filename, label, functionDescriptor):
    features= []

    for c in filename:
        features.append(functionDescriptor(c))
        
#     features= np.array(features)
#     label=np.array(label)
#     dFeatures = tf.data.Dataset.from_tensor_slices(features)
#     dLabels = tf.data.Dataset.from_tensor_slices(label)
#     dataset = tf.data.Dataset.zip((dFeatures,dLabels)).shuffle(500).batch(50)

    dataset = list(zip(features,label))
    random.shuffle(dataset)
    features,label = list(zip(*dataset))

    dataset = np.array([features,label])

    return dataset
    
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