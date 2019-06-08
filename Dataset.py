import cv2
import os
import numpy as np
import random


class Data(object):
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
    
    def _CNNloadImage(self,imgPath):
        img = cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (100, 100)) 
        img /=255.0
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
