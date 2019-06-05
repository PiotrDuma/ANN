# import os,sys
import numpy as np
import functions
import tensorflow as tf
import matplotlib.pyplot as plt

class NNmodel:
    #  Parameters
    learningRate = 0.1
    epochs = 200
#     batchSize = 100
    displayStep = 1
       
 
    def __init__(self, hidden1=256, hidden2=256, location = "data", desc = functions.getHOGDescriptor):
        self.numHidden1 = hidden1
        self.numHidden2 = hidden2
        self.numInput=0   #size of input vector
        self.numClasses=0 #size of output (classes) vector
        self.activeDescriptor = desc
        
        self.mypath = location         
        #dataset variables
        self.DATASET= functions.Data(self.mypath) #dataset class object
        self.training=None
        self.validation=None
        self.test=None
        
        #Network session variables
        self._session=None #network session
        #network vectors: weights, biases
        self.weights=None
        self.biases=None
        self._x=None
        self._w=None
        self._b=None
        self._y=None
        
        #preinitialize numpy output tables ~ dont need to copy every epoch.
        self.costHistory = np.zeros([self.epochs,1], dtype=float)
        self.accuracy = np.zeros([self.epochs,1], dtype=float)
        self.error =np.zeros([self.epochs,1], dtype=float)
        self.classificationResults=None
        self.classificationTable = None
        
    def prepareDataset(self, ref2Descriptor = None):
        if ref2Descriptor == None:
            ref2Descriptor = self.activeDescriptor
            
        self.DATASET.preprocessing(ref2Descriptor) #create Data object
        self.training= self.DATASET.getTrainingSet()
        self.validation= self.DATASET.getValidationSet()
        self.test= self.DATASET.getTestingSet()
        
        self.numInput = len(self.training[0][0]) #input shape
        self.numClasses = len(self.training[1][0]) #output classes
        
    def initNetworkVariables(self):
        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([self.numInput,self.numHidden1])),
            'h2': tf.Variable(tf.truncated_normal([self.numHidden1,self.numHidden2])),
            'out': tf.Variable(tf.truncated_normal([self.numHidden2,self.numClasses]))
            }
        self.biases = {
            'b1': tf.Variable(tf.truncated_normal([self.numHidden1])),
            'b2': tf.Variable(tf.truncated_normal([self.numHidden2])),
            'out': tf.Variable(tf.truncated_normal([self.numClasses]))
            }
        self._x = tf.placeholder(tf.float32, [None, self.numInput])
        self._w = tf.Variable(tf.zeros([self.numInput,self.numClasses]))
        self._b = tf.Variable(tf.zeros([self.numClasses]))
        self._y = tf.placeholder(tf.float32, [None, self.numClasses])
        
    def process(self,x, weights, biases):
        layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer1 = tf.nn.sigmoid(layer1)
         
        layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
        layer2 = tf.nn.sigmoid(layer2)
     
        outlayer = tf.matmul(layer2, weights['out']) + biases['out']
        return outlayer
    
    def start(self):
        self.initNetworkVariables()
        init = tf.global_variables_initializer()
        
        #network model settings
        pred = self.process(self._x, self.weights, self.biases) #function with placeholder
        costFunction = tf.reduce_mean(-tf.reduce_sum(self._y*tf.log(tf.nn.softmax(pred)),1))
        correctPred = tf.equal(tf.argmax(pred,1), tf.argmax(self._y,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred,tf.float32))
        trainingStep = tf.train.GradientDescentOptimizer(self.learningRate).minimize(costFunction)
        ###
        
        self._session = tf.Session()
        self._session.run(init)
      
        for epoch in range(self.epochs):
            #training part
            self.training = self.shuffleNumpyXY(self.training)
            feedDict = {self._x: self.training[0], self._y: self.training[1] }
            self._session.run(trainingStep, feed_dict = feedDict)
            self.costHistory[epoch] = self._session.run(costFunction, feed_dict =feedDict)

            predY = self._session.run(pred, feed_dict={self._x: self.training[0]})
            mse = tf.reduce_mean(tf.square(predY - self.training[1]))
            
            self.error[epoch] = self._session.run(mse)
            #validation part
            self.accuracy[epoch] = (self._session.run(accuracy, feed_dict={self._x: np.vstack(self.validation[0]), self._y: np.vstack(self.validation[1])}))
            if epoch % self.displayStep==0:
                print('epoch : ', epoch, ' cost: ', self.costHistory[epoch], "  error: ", self.error[epoch], " Acc: ", self.accuracy[epoch])
        
    def plot(self):
        plt.plot(self.error, 'r')
        plt.show()
        
        plt.plot(self.costHistory, 'g')
        plt.show()
           
        plt.plot(self.accuracy,'b')
        plt.show()
        
    def shuffleNumpyXY(self, table):
        assert(len(table[0]) == len(table[1]))
        p = np.random.permutation(len(table[1]))   
        return [table[0][p], table[1][p]]
        
    def runTest(self):
        self.classificationResults= np.zeros(len(self.test[1][0]),dtype=np.int16)
        self.classificationTable = np.zeros([len(self.test[1][0]),len(self.test[1][0])],dtype=np.int16) #YxY table, [class, classified as]
        
        ##calculate propagation
        pred = self.process(self._x, self.weights, self.biases) #placeholder
        resultsX = self._session.run(pred, feed_dict={self._x: self.test[0]})
        
        for i in range(len(resultsX)):
            if(np.argmax(resultsX[i]) == np.argmax(self.test[1][i])): #predict == Y
                self.classificationResults[np.argmax(self.test[1][i])] +=1
        
            self.classificationTable[np.argmax(self.test[1][i])][np.argmax(resultsX[i])]+=1 ## [realY][predictedY] +=1
          
        print("TEST DONE.")
        print(self.DATASET.getUniqueLabels())
        ##check correctness
        for i in range(len( self.classificationResults)):
            if(self.classificationResults[i]!=self.classificationTable[i][i]):
                print("TABLE CLASSIFICATION ERROR")
                print(self.classificationResults)
                print("table")
                print(self.classificationTable)
                
        print("ACCURACY RESULTS: ",self.classificationResults)
        print("ACCURACY TABLE")
        print(self.classificationTable)
        
        classify = np.multiply(self.classificationResults, 1/(len(resultsX)/len(self.test[1][0])))
        print("%%: ", classify)  
        
    def saveTestSeries(self):

        tmp = "\nMLP " +self.activeDescriptor.__name__+" "+self.numHidden1.__str__()+", "+self.numHidden2.__str__()+" "+self.mypath+" "+"\n" 
        with open("OUTPUT.txt", 'a') as file:
            file.write(tmp)

            file.write("costHistory:\n")
            np.savetxt(file,self.costHistory, newline=',')

            file.write("\naccuracy:\n")
            np.savetxt(file,self.accuracy, newline=',')

            file.write("\nError:\n")
            np.savetxt(file,self.error, newline=',')
            file.write("\n" )

            for word in self.DATASET.getUniqueLabels():
                file.write(word + ', ')
            file.write("\nResults:\n" )
            for word in self.classificationResults:
                file.write(word.__str__() + ', ')
            file.write("\nTABLE\n" )
            np.savetxt(file,self.classificationTable,fmt = '%5i')
            file.write("\n* * * * * * * * * * * * * * * *\n" )
            


if __name__ == "__main__":
    myDescriptors =[functions.getHOGDescriptor ]
    mynetwork = NNmodel(64,64, "data2",myDescriptors[0])
    mynetwork.prepareDataset()
    print("training set length: ",len(mynetwork.training[0]))
    print("validation set length: ",len(mynetwork.validation[0]))
    print("testing set length: ",len(mynetwork.test[0]))
    mynetwork.start()
    mynetwork.plot()
    mynetwork.runTest()
    mynetwork.saveTestSeries()
