import tensorflow as tf
class ModelShape:

    @staticmethod
    def _modelShape(features, mode, numClasses):

        x= features["x"]
#         inputLayer = tf.reshape(x, self._fitPlaceholder())
        inputLayer = tf.reshape(x,[-1, 100, 100, 3])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=inputLayer,name="layer1",filters=32,kernel_size=[5,5],
                                 padding="same",activation=tf.nn.relu)
        
                
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1,name="layer1", pool_size=2, strides=2)
        
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(inputs=pool1,filters=64,name="layer3",kernel_size=[5,5],
                                 padding="same",activation=tf.nn.relu)
        
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2,name="layer4", pool_size=2, strides=2)
        
        #flatten to 1d array
        flatten = tf.contrib.layers.flatten(pool2)
        
        #fully connected
        dense = tf.layers.dense(inputs=flatten, name='dense1',units=500, activation=tf.nn.relu)
        
        dropout = tf.layers.dropout(dense, 0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
#         output = tf.layers.dense(inputs=dense, name='dense2',units=self.numClasses)
        output = tf.layers.dense(inputs=dropout, name='dense2',units=numClasses)
      
        return output

    @staticmethod
    def _modelShapeClassic(features, mode, numClasses):

        x= features["x"]
#         inputLayer = tf.reshape(x, self._fitPlaceholder())
        inputLayer = tf.reshape(x,[-1, 100, 100, 3])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=inputLayer,name="layer1",filters=32,kernel_size=[4,4],
                                 padding="same",activation=tf.nn.relu)
        
                
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1,name="layer1", pool_size=2, strides=2)
        
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(inputs=pool1,filters=64,name="layer3",kernel_size=[4,4],
                                 padding="same",activation=tf.nn.relu)
        
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2,name="layer4", pool_size=2, strides=2)
        #flatten to 1d array
        flatten = tf.contrib.layers.flatten(pool2)
        
        #fully connected
        dense = tf.layers.dense(inputs=flatten, name='dense1',units=250, activation=tf.nn.relu)
        
        dropout = tf.layers.dropout(dense, 0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
#         output = tf.layers.dense(inputs=dense, name='dense2',units=self.numClasses)
        output = tf.layers.dense(inputs=dropout, name='dense2',units=numClasses)
      
        return output
    
    @staticmethod
    def _modelShapeKPPK(features, mode, numClasses):

        x= features["x"]
#         inputLayer = tf.reshape(x, self._fitPlaceholder())
        inputLayer = tf.reshape(x,[-1, 100, 100, 3])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=inputLayer,name="layer1",filters=32,kernel_size=[5,5],
                                 padding="same",activation=tf.nn.relu)
        
                
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1,name="layer2", pool_size=2, strides=2)
        
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=pool1,name="layer3", pool_size=2, strides=2)
        
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(inputs=pool2,filters=64,name="layer4",kernel_size=[4,4],
                                 padding="same",activation=tf.nn.relu)
    
        
        #flatten to 1d array
        flatten = tf.contrib.layers.flatten(conv2)
        
        #fully connected
        dense = tf.layers.dense(inputs=flatten, name='dense1',units=250, activation=tf.nn.relu)
        
        dropout = tf.layers.dropout(dense, 0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
#         output = tf.layers.dense(inputs=dense, name='dense2',units=self.numClasses)
        output = tf.layers.dense(inputs=dropout, name='dense2',units=numClasses)
      
        return output
    
    @staticmethod
    def _modelShapeKKK(features, mode, numClasses):

        x= features["x"]
#         inputLayer = tf.reshape(x, self._fitPlaceholder())
        inputLayer = tf.reshape(x,[-1, 100, 100, 3])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=inputLayer,name="layer1",filters=32,kernel_size=[5,5],
                                 padding="same",activation=tf.nn.relu)
        
        
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(inputs=conv1,filters=32,name="layer2",kernel_size=[4,4],
                                 padding="same",activation=tf.nn.relu)
    
        # Convolutional Layer #2
        conv3 = tf.layers.conv2d(inputs=conv2,filters=32,name="layer3",kernel_size=[2,2],
                                 padding="same",activation=tf.nn.relu)
        #flatten to 1d array
        flatten = tf.contrib.layers.flatten(conv3)
        
        #fully connected
        dense = tf.layers.dense(inputs=flatten, name='dense1',units=250, activation=tf.nn.relu)
        
        dropout = tf.layers.dropout(dense, 0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
#         output = tf.layers.dense(inputs=dense, name='dense2',units=self.numClasses)
        output = tf.layers.dense(inputs=dropout, name='dense2',units=numClasses)
      
        return output
    
    @staticmethod
    def _modelShapePKK(features, mode, numClasses):

        x= features["x"]
#         inputLayer = tf.reshape(x, self._fitPlaceholder())
        inputLayer = tf.reshape(x,[-1, 100, 100, 3])

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=inputLayer,name="layer2", pool_size=2, strides=2)

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=pool1,name="layer1",filters=32,kernel_size=[5,5],
                                 padding="same",activation=tf.nn.relu)
        
        
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(inputs=conv1,filters=64,name="layer4",kernel_size=[5,5],
                                 padding="same",activation=tf.nn.relu)
    
        #flatten to 1d array
        flatten = tf.contrib.layers.flatten(conv2)
        
        #fully connected
        dense = tf.layers.dense(inputs=flatten, name='dense1',units=250, activation=tf.nn.relu)
        
        dropout = tf.layers.dropout(dense, 0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
#         output = tf.layers.dense(inputs=dense, name='dense2',units=self.numClasses)
        output = tf.layers.dense(inputs=dropout, name='dense2',units=numClasses)
      
        return output