import os,sys
import numpy as np
import functions
import tensorflow as tf
import matplotlib.pyplot as plt


def process(x, weights, biases):
    
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)
        
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)
    
    outlayer = tf.matmul(layer2, weights['out']) + biases['out']
    
    return outlayer

#START 
mypath = "data"
imagePath = os.path.join(sys.path[0], mypath)

print("Loading images")
# DATASET, validation = functions.preprocessing(imagePath,functions.getHOGDescriptor)
DATASET, validation = functions.preprocessing(imagePath,functions.getHOGDescriptor)

# split testing and validate data to input and output lists
trainX = np.vstack(DATASET[0])
trainY = np.vstack(DATASET[1])
testX = np.vstack(validation[0])
testY = np.vstack(validation[1])

print(trainX.shape)
print(trainY.shape)

#  Parameters
learningRate = 0.1
epochs = 200
batchSize = 100
displayStep = 100
  
# Network Parameters
numHidden1 = 256 # 1st layer number of neurons
numHidden2 = 256# 2nd layer number of neurons

numInput = len(DATASET[0][0]) #input shape
numClasses = len(DATASET[1][1]) #output classes
  
x = tf.placeholder(tf.float32, [None, numInput])
w = tf.Variable(tf.zeros([numInput, numClasses]))
b = tf.Variable(tf.zeros([numClasses]))
y = tf.placeholder(tf.float32, [None, numClasses])

weights = {
    'h1': tf.Variable(tf.truncated_normal([numInput,numHidden1])),
    'h2': tf.Variable(tf.truncated_normal([numHidden1,numHidden2])),
    'out': tf.Variable(tf.truncated_normal([numHidden2,numClasses]))
    }

biases = {
    'b1': tf.Variable(tf.truncated_normal([numHidden1])),
    'b2': tf.Variable(tf.truncated_normal([numHidden2])),
    'out': tf.Variable(tf.truncated_normal([numClasses]))
    }

init = tf.global_variables_initializer()

saver=tf.train.Saver()

pred = process(x, weights, biases)

# crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pred,logits=y)
# costFunction = tf.reduce_mean(crossEntropy)
# costFunction = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits = y, labels = pred))
costFunction = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.nn.softmax(pred)),1))

trainingStep = tf.train.GradientDescentOptimizer(learningRate).minimize(costFunction)


sess = tf.Session()
sess.run(init)
# with tf.Session() as sess:
#     sess.run(init)

error = []
accuracyHist = []
costHist = np.empty(shape=[1], dtype=float)

for epoch in range(epochs):
    sess.run(trainingStep, feed_dict = {x: trainX, y: trainY })
    cost = sess.run(costFunction, feed_dict = {x: trainX, y: trainY })
    costHist = np.append(costHist, cost)
    
    correctPred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred,tf.float32))

    predY = sess.run(pred, feed_dict={x: testX})
    mse = tf.reduce_mean(tf.square(predY - testY))
    
    mseErr = sess.run(mse)
    error.append(mseErr)
    
    accuracy = (sess.run(accuracy, feed_dict={x: trainX, y: trainY}))
    accuracyHist.append(accuracy)
    
    print('epoch : ', epoch, ' cost: ', cost, "  error: ", mseErr, " Acc: ", accuracy)

saver.save(sess, sys.path[0])

plt.plot(error, 'r')
plt.show()
 
plt.plot(accuracyHist)
plt.show()

# print(accuracyHist)


####CLASSIFICATION RESULTS

classificationResults = np.zeros(len(testY[0]))
print("TESTS")

##calculate propagation
resultsX = sess.run(pred, feed_dict={x: testX})

for i in range(len(resultsX)):
    if(np.argmax(resultsX[i]) == np.argmax(testY[i])):
        classificationResults[np.argmax(testY[i])] +=1


print("SHAPE: ",classificationResults)

classify = np.multiply(classificationResults, 1/(len(resultsX)/len(testY[0])))


names = sorted(os.walk(imagePath).__next__()[1])
# print(names)

tmp = np.arange(len(names))
plt.bar(tmp, classify)
plt.xticks(tmp, names,rotation=45)
plt.show()

# 
# with open("error_histogram.txt", 'w') as f:
#     for s in error:
#         f.write(str(s) + '\n')
# 


np.savetxt('acc_hist_2.txt', accuracyHist, delimiter='\n')   # X is an array


