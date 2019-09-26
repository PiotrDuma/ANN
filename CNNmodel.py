import sys
import numpy as np
from Dataset import Data
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import shape
import cv2
from ModelShapes import ModelShape

class CNNmodel():
    #  Parameters
#     learningRate = 0.001
    epochs = 30
    batchSize = 100
    displayStep = 1
       
    def __init__(self,location = "data", modelDir="./model/test", shape= ModelShape._modelShape, LRate=0.001):
        self.learningRate =LRate
        self.numInput=0   #size of input vector
        self.numClasses=0 #size of output (classes) vector
        
        self.mypath = location
        self.modelDir = modelDir  
        self.modelShape = shape
        #dataset variables
        self.DATASET= Data(self.mypath) #dataset class object
        self.training=None
        self.validation=None
        self.test=None
        self.trainingDataset = None
        

        self._session = None
        self.cnnEstimator = None
        
        self.costHistoryTrain = np.zeros([self.epochs,1], dtype=float)
        self.accuracyTrain = np.zeros([self.epochs,1], dtype=float)
        
        self.costHistoryValid = np.zeros([self.epochs,1], dtype=float)
        self.accuracyValid = np.zeros([self.epochs,1], dtype=float)
        self.classificationResults=None
        self.classificationTable = None
        
    @staticmethod
    def _CNNloadImage(imgPath):
        img = cv2.imread(imgPath,cv2.IMREAD_UNCHANGED)
#         img = cv2.imread(imgPath,cv2.COLOR_BGR2GRAY)
        if img is not None:
            img = cv2.resize(img, (100, 100)) 
            img = (img/255.0)
        else:
            print("Load image error. Image is None."+imgPath)
        return [img]
    
    def loadDataset(self):      
        self.DATASET.preprocessing(CNNmodel._CNNloadImage) #create Data object
        print("check  ", type(self.DATASET.getTrainingSet()))
        
#         self.training= self.DATASET.getTrainingSet()
#         self.validation= self.DATASET.getValidationSet()
#         self.test= self.DATASET.getTestingSet()
        #type translate required
        self.training= list(map(lambda x:x.astype(np.float32) ,self.DATASET.getTrainingSet()))
        self.validation= list(map(lambda x:x.astype(np.float32) ,self.DATASET.getValidationSet()))
        self.test= list(map(lambda x:x.astype(np.float32) ,self.DATASET.getTestingSet()))

         
        print("check1  ", type(self.training))
        print("check2  ", self.test[0].dtype)
        print("check2  ", self.test[1].dtype)
        print("check3  ", type(self.training[1]))
        self.numInput = len(self.training[0]) #input shape
        self.numClasses = len(self.training[1][0]) #output classes
        print(shape(self.training[0]))
        print(shape(self.training[1]))

     
    def _CNNmodelFunction(self,features, labels, mode):
        
        
        #produce output through network model; stored in ModelShapes.py
        output = self.modelShape(features, mode, self.numClasses)
        
        #predictions clases
        predClass = tf.argmax(output,axis=1)
        
        
#         #training prediction probability
#         predProb = tf.nn.softmax(logits=output, name= "softmax_tensor")

                
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predClass)
        else:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output) 
#             cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels= labels, logits=output)
            
            #count loss  based on function
            loss = tf.reduce_mean(cross_entropy,name='loss')
    
            # Define the optimizer for improving the neural network.
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
    
            # Get the TensorFlow op for doing a single optimization step.
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
            # Define the evaluation metrics,
            # in this case the classification accuracy.

            accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),predictions=tf.argmax(output,1))
            tf.summary.scalar('accuracy', accuracy[1])
            metrics = {'accuracy': accuracy}
            
            
            # Wrap all of this in an EstimatorSpec.  
            if mode == tf.estimator.ModeKeys.EVAL:
                spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss,eval_metric_ops= metrics)
            else:
                spec = tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,eval_metric_ops= metrics)
                
        return spec


    def start(self):
           
        self.cnnEstimator= tf.estimator.Estimator(model_fn=self._CNNmodelFunction, model_dir=self.modelDir)
 

        trainInput = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.training[0]},
            y=self.training[1],
#             batch_size=self.batchSize,
            num_epochs=1,
            shuffle=True)
        
        validationInput = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.validation[0]},
            y=self.validation[1],
            num_epochs=1,
            shuffle=False)
            
        print("START TRAINING")
        
     
        #### ESTIMATOR IN MEMORY HOOK EVALUATOR   ~   DOESN'T WORK AT THIS MOMENT
        #### evaluate base during training. Err: dict value of loss and acc is 0. 
        #### https://github.com/tensorflow/tensorflow/issues/21590
# #         evaluator = tf.estimator.experimental.InMemoryEvaluatorHook(estimator=self.cnnEstimator,input_fn=validationInput,every_n_iter=1  )
# #         tf.logging.set_verbosity(tf.logging.INFO)
# #         self.cnnEstimator.train(input_fn=trainInput, hooks=[evaluator], steps=self.epochs)
# #   
# # #         self.cnnEstimator.train(trainInput, steps=self.epochs)
# #         print(self.cnnEstimator.evaluate(input_fn=validationInput, steps=1))
        ########################################################################

        #### ONE EPOCH METHOD
        #### TRAIN THEN EVALUATE EVERY EPOCH
        #### it's not optimal way to train ANN, but project assumptions are required to store data 
        #### (trained and evaluated) every single epoch, not with the period of time
         
        for i in range(self.epochs):
            self.cnnEstimator.train(input_fn=trainInput,steps=1)#,hooks=[evaluator] 
            valueTrain = self.cnnEstimator.evaluate(input_fn=trainInput, steps=1)
            valueValid = self.cnnEstimator.evaluate(input_fn=validationInput, steps=1)#, hooks=[evaluator]
            self.accuracyTrain[i] = valueTrain['accuracy']
            self.costHistoryTrain[i] = valueTrain['loss']
            self.accuracyValid[i] = valueValid['accuracy']
            self.costHistoryValid[i] = valueValid['loss']
             
            print("train: ", valueTrain)
             
        #### END OF ONE EPOCH METHOD ####

        
    def predict(self, set):
        self.classificationResults= np.zeros(len(set[1][0]),dtype=np.int16)
        self.classificationTable = np.zeros([len(set[1][0]),len(set[1][0])],dtype=np.int16) #YxY table, [class, classified as]
        
        testInput = tf.estimator.inputs.numpy_input_fn(
            x={"x": set[0]},
            num_epochs=1,
            shuffle=False)
        
        predictions= self.cnnEstimator.predict(testInput)
        
        for i, result in enumerate(predictions):
            if(set[1][i][result]==1):
                self.classificationResults[result] +=1
            self.classificationTable[np.argmax(set[1][i])][result]+=1 ## [realY][predictedY] +=1
                
        print("TEST DONE.")
        print(self.DATASET.getUniqueLabels())
        ##check correctness
        for i in range(len( self.classificationResults)):
            if(self.classificationResults[i]!=self.classificationTable[i][i]):
                print("TABLE CLASSIFICATION ERROR")
                print(self.classificationResults)
                print("table")
                print(self.classificationTable)
                
        print("ACCURACY RESULTS: ",self.classificationResults, "  %: ", (sum(self.classificationResults)/len(set[1])))
        print("ACCURACY TABLE")
        print(self.classificationTable)


        classify = np.multiply(self.classificationResults, 1/(len(set[1])/len(set[1][0])))
        print("%%: ", classify)  
        
        
    def saveTestSeries(self, filename="output.txt"):
        tmp = "\nCNN " +self.mypath+" "+"learningRate:"+str(self.learningRate)+"\n" 
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

            file.write("\n" )


            for word in self.DATASET.getUniqueLabels():
                file.write(word + ', ')
            file.write("\nResults:  overall: " + str((sum(self.classificationResults)/len(self.test[0]))) +"\n" )
            for word in self.classificationResults:
                file.write(word.__str__() + ', ')
            file.write("\nTABLE\n" )
            np.savetxt(file,self.classificationTable,fmt = '%5i')
            file.write("\n* * * * * * * * * * * * * * * *\n" )
        
    def plot(self):
      
        plt.plot(self.costHistoryTrain, 'g')
        plt.plot(self.costHistoryValid, 'k')
        plt.show()
         
        plt.plot(self.accuracyTrain,'b')
        plt.plot(self.accuracyValid, 'k')
        plt.show()
        
if __name__ == "__main__":
    
    modelShapes = [ModelShape._modelShape, ModelShape._modelShapeClassic, ModelShape._modelShapeKPPK, ModelShape._modelShapePKK, ModelShape._modelShapeKKK ]
    DB = ["dataset1","dataset2","dataset3"]
    learningRate = [0.01, 0.001, 0.0001]
    
    
    if(len(sys.argv) == 5):       
        dataset = int(sys.argv[1])
        modelShape = int(sys.argv[2])
        nLoop = int(sys.argv[3])
        LRate = int(sys.argv[4])
           
        modelName = DB[dataset] +"_" + modelShapes[modelShape].__name__ +"_Out250_LR_SIGMOID_" + str(learningRate[LRate]).replace(".", "")
           
        mynetwork = CNNmodel(DB[dataset],"./model/"+modelName+ "_iter_"+str(nLoop), modelShapes[modelShape],learningRate[LRate])
    else:
        
        modelName = "dataset1__modelShapeKPPK_Out250_LR0001_"
        mynetwork = CNNmodel(DB[0],"./model/"+modelName, modelShapes[2], learningRate[1])
       
       
    mynetwork.loadDataset()
    mynetwork.start()
    
    mynetwork.predict(mynetwork.test)
    mynetwork.saveTestSeries("./OUTPUT/" +modelName +".txt")
    mynetwork.plot()
    
    pass