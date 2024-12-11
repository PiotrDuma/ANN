# ANN

Artificial neural networks for image classification based on Multilayer Perceptron (MLP) and Convolutional Neural Networks (CNN) models. Three databases of fruit images are prepared for training and validation datasets, which are improved by white noise and rotation to keep its consistency and increase distinguishability at the same time. The availability of various image descriptors and models allows to study effectiveness and efficiency of neural networks, classification accuracy, and learning process.

## Caution

1. Project is running on Tensorflow 1.x version, which is actually deprecated.
2. All image descriptors have been implemented in research purposes. Some of them like histogram of colours or local binary pattern may contain lack of desired information required to successfully train neural network. In other words, not every method of image description works in terms of MLP input data.  

## Instalation

This project works under Python 3.6.8 version, which has to be forced due to deprecated interfaces of later released frameworks.

### Enviroment

1. Install Python 3.6.8 [[download]](https://www.python.org/downloads/release/python-368/) and add it to your system variables [[guide]](https://realpython.com/add-python-to-path/).
2. Install fallowing liblaries, fallow the commands in your system console
+ tensoflow v1.14 [[doc]](https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs)
```
python -m pip install tensorflow==1.14
```
+ matplotlib v3.0.3
```
python -m pip install matplotlib==3.0.3
```
+ numpy v1.16.4
```
python -m pip install numpy==1.16.4
```
+ opencv-python v3.4.5.20
```
python -m pip install opencv-python 3.4.5.20
```
+ scikit-image v0.15.0
```
python -m pip install scikit-image==0.15.0
```
+ scipy v1.5.4
```
python -m pip install scipy==1.5.4
```

**Make sure installed liblaries are up to those versions. Sometimes installing them via pip updates dependencies.** Instalation of opencv liblary automaticly downloads numpy with higher version. In that case, uninstall numpy and install valid release.
   
### Clone project

1. Open your console and navigate to parent directory
2. Clone github repository
```
git clone https://github.com/PiotrDuma/ANN.git
```
3. Navigate to following directory. Python project is ready to run.

## Run

There are two different entry points to the project that can be invoked via neural network architecture classes, NNmodel.py for MLP networks and CNNmodel.py for convolutional networks, respectively.

### MLP

a) console specified parameters

To automate the process of running the project mutliple times, console arguments have been added. Program could be run using combination of preprogrammed input values in RUN_TEST_SCRIPT.bat windows script or type the command written down below:

```
python NNmodel.py x y z i
```
where:
x - is the table index[0-2] of database:
```
227    DB = ["dataset1","dataset2","dataset3"]
```
y - is the table index[0-3] of image descriptor:
```
226    myDescriptors = [dsc.getHOGDescriptor, dsc.getLocalBinaryPatterns, dsc.getHistogram, dsc.mix]
```
z - is the table index[0-7] of layer size combination:
```
229       sizes = [(50,50),(100,100), (150,150), (200,200), (250,250),(200,100),(100,200),(500,500)]
```
z - is the table index[0-2] of learning rate:
```
230    learningRate = [0.01, 0.001, 0.0001]
```

b) incode specified parameters
If the running call has number of arguments other than 4, then program will run with parameter values declared in code.

```
python NNmodel.py
```

There's much more flexibility to specify layer size, learning rate in class declaration(line 251). For advanced users, there's possibility to declare custom descriptors or change model parameters like function of neuron activation, learning algorithm or values of neurons' connection weights and biases. Feel free to explore. Tensorflow [documentation](https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs) might be helpful.

## Dataset

There are three different databases containing pictures of fruits:

1. Images of objects in the centre of picture with removed background.
  source: https://github.com/Horea94/Fruit-Images-Dataset/tree/3571f04df801a3c09c24905f4eccbc159dbab60c/Training

2. Pictures of fruits with complex background. Simple image has only one object in different place. Objects might be deformed due to image resizing and rotating process. 
  source: my own pictures multiplied with image editing 
  
3. Probably the hardsest problem of difficulty level. Images contains any number of fruits located randomply over the picture.
  source: http://doi.org/10.5281/zenodo.1310165
