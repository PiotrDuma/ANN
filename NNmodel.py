import sys
import numpy as np
import Dataset
from Descriptor import Descriptor as dsc
import tensorflow as tf
import matplotlib.pyplot as plt

class NNmodel(object):
    #  Parameters
    
    epochs = 200
#     batchSize = 100
    displayStep = 1
       
 
    def __init__(self, hidden1=256, hidden2=256, location = "data", desc = dsc.getHOGDescriptor, LR=0.001):
        self.learningRate = LR
        self.numHidden1 = hidden1
        self.numHidden2 = hidden2
        self.numInput=0   #size of input vector
        self.numClasses=0 #size of output (classes) vector
        self.activeDescriptor = desc
        
        self.mypath = location         
        #dataset variables
        self.DATASET= Dataset.Data(self.mypath) #dataset class object
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
        self.costHistoryTrain = np.zeros([self.epochs,1], dtype=float)
        self.accuracyTrain = np.zeros([self.epochs,1], dtype=float)
        self.errorTrain =np.zeros([self.epochs,1], dtype=float)
        self.classificationResults=None
        self.classificationTable = None
        
        self.costHistoryValid = np.zeros([self.epochs,1], dtype=float)
        self.accuracyValid = np.zeros([self.epochs,1], dtype=float)
#         self.errorValid =np.zeros([self.epochs,1], dtype=float)
        
        
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

        
        #network model settings
        pred = self.process(self._x, self.weights, self.biases) #function with placeholder
#         costFunction = tf.reduce_mean(-tf.reduce_sum(self._y*tf.log(tf.nn.softmax(pred)),1))
        costFunction=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred, labels =self._y))
        #sigmoid 
#         costFunction=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels =self._y))  
        
        
        maxPred = tf.nn.softmax(pred)
        correctPred = tf.equal(tf.argmax(maxPred,1), tf.argmax(self._y,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred,tf.float32))
#         trainingStep = tf.train.GradientDescentOptimizer(self.learningRate).minimize(costFunction)
        trainingStep = tf.train.AdamOptimizer(learning_rate = self.learningRate).minimize(loss=costFunction)
        ###
        
        init = tf.global_variables_initializer()
        
                
        self._session = tf.Session()
        self._session.run(init)
      
        for epoch in range(self.epochs):
            #training part
            self.training = self.shuffleNumpyXY(self.training)
            feedDict = {self._x: self.training[0], self._y: self.training[1] }
            self._session.run(trainingStep, feed_dict = feedDict)

            ##### training data #####
            self.costHistoryTrain[epoch] = self._session.run(costFunction, feed_dict =feedDict)

            predY = self._session.run(pred, feed_dict={self._x: self.training[0]})
            mse = tf.reduce_mean(tf.square(predY - self.training[1]))
            
            #loss function as error
            self.errorTrain[epoch] = self._session.run(mse)
            
            #acc
#             self.accuracyTrain[epoch] = (self._session.run(accuracy, feed_dict={self._x: np.vstack(self.training[0]), self._y: np.vstack(self.training[1])}))
            self.accuracyTrain[epoch] = (self._session.run(accuracy, feed_dict={self._x: self.training[0], self._y: self.training[1]}))
           
            
            ######  validation data   ####
            self.costHistoryValid[epoch] = self._session.run(costFunction, feed_dict ={self._x: self.validation[0], self._y: self.validation[1] })
#             self.errorValid[epoch] = self._session.run(tf.reduce_mean(tf.square(predY - self.validation[1])))
            #validation part
#             self.accuracyValid[epoch] = (self._session.run(accuracy, feed_dict={self._x: np.vstack(self.validation[0]), self._y: np.vstack(self.validation[1])}))
            self.accuracyValid[epoch] = (self._session.run(accuracy, feed_dict={self._x: self.validation[0], self._y: self.validation[1]}))
            #display
            if epoch % self.displayStep==0:
#                 print('epoch : ', epoch, ' cost: ', self.costHistoryValid[epoch], "  error: ", self.errorTrain[epoch], " Acc: ", self.accuracyValid[epoch])
                print('epoch : ', epoch, ' cost: ', self.costHistoryTrain[epoch],self.costHistoryValid[epoch], "  error: ", self.errorTrain[epoch], " Acc: ",  self.accuracyTrain[epoch],self.accuracyValid[epoch])
    
    def plot(self):
      
        plt.plot(self.costHistoryTrain, 'g')
        plt.plot(self.costHistoryValid, 'k')
        plt.show()
         
        plt.plot(self.accuracyTrain,'b')
        plt.plot(self.accuracyValid, 'k')
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
                
        print("ACCURACY RESULTS: ",self.classificationResults, "  %: ", (sum(self.classificationResults)/len(resultsX)))
        print("ACCURACY TABLE")
        print(self.classificationTable)
        
        classify = np.multiply(self.classificationResults, 1/(len(resultsX)/len(self.test[1][0])))
        print("%%: ", classify)  
        
    def saveTestSeries(self, filename="output.txt"):
        tmp = "\nMLP " +self.activeDescriptor.__name__+" "+self.numHidden1.__str__()+", "+self.numHidden2.__str__()+" "+self.mypath+" "+"learningRate:"+str(self.learningRate)+"\n" 
        with open(filename, 'a') as file:
            file.write(tmp)

            file.write("costHistoryTrain:\n")
            np.savetxt(file,self.costHistoryTrain, newline=',')

            file.write("\naccuracyTrain:\n")
            np.savetxt(file,self.accuracyTrain, newline=',')

            file.write("\ncostHistoryValid:\n")
            np.savetxt(file,self.costHistoryValid, newline=',')

            file.write("\naccuracyValid:\n")
            np.savetxt(file,self.accuracyValid, newline=',')

#             file.write("\nErrorValid:\n")
#             np.savetxt(file,self.errorValid, newline=',')
            file.write("\n" )


            for word in self.DATASET.getUniqueLabels():
                file.write(word + ', ')
            file.write("\nResults:  overall: " + str((sum(self.classificationResults)/len(self.test[0]))) +"\n" )
            for word in self.classificationResults:
                file.write(word.__str__() + ', ')
            file.write("\nTABLE\n" )
            np.savetxt(file,self.classificationTable,fmt = '%5i')
            file.write("\n* * * * * * * * * * * * * * * *\n" )
            


if __name__ == "__main__":

    myDescriptors = [ dsc.getHOGDescriptor, dsc.getLocalBinaryPatterns, dsc.getHistogram ,dsc.mix ]
    DB = ["dataset1","dataset2","dataset3"]

    sizes = [(50,50),(100,100), (150,150), (200,200), (250,250),(200,100),(100,200),(500,500)]
    learningRate = [0.01, 0.001, 0.0001]
    
    
    #run powershell script (index: 1 - dataset, 2 - descriptor, 3 - architecture, 4 - learning rate)
    if(len(sys.argv) == 5):
        dataset = int(sys.argv[1])
        dsc = int(sys.argv[2])
        size = int(sys.argv[3])
        LR = int(sys.argv[4])
        print("Starting: base: ", DB[dataset], "  descriptor:  ", myDescriptors[dsc].__name__, "  size:  ",str(sizes[size])   )        
      
        mynetwork = NNmodel(sizes[size][0],sizes[size][1], DB[dataset],myDescriptors[dsc], learningRate[LR])
        mynetwork.prepareDataset()
           
        mynetwork.start()
       
        outputName = "./OUTPUT/" +DB[dataset]+ "_"+ myDescriptors[dsc].__name__+ "_"+str(sizes[size][0])+ "_"+str(sizes[size][1])+ "_NOISED_LR"+str(learningRate[LR]) +".txt"
        mynetwork.runTest()
        mynetwork.saveTestSeries(outputName)
        
    else: #default run options
        mynetwork = NNmodel(250,250, "dataset1",myDescriptors[0], 0.001)
        mynetwork.prepareDataset()
        print("training set length: ",len(mynetwork.training[0]))
        print("validation set length: ",len(mynetwork.validation[0]))
        print("testing set length: ",len(mynetwork.test[0]))
           
        print("length descriptor x: ",len(mynetwork.training[0][0]))
       
        mynetwork.start()
          
        mynetwork.runTest()
        mynetwork.saveTestSeries("./OUTPUT/TESTRUN_dataset3_HOG_250_250_LR_0001.txt")
        mynetwork.plot()
        

    pass