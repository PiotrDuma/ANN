# ANN

Artificial neural networks for image classification based on Multilayer Perceptron (MLP) and Convolutional Neural Networks (CNN) models. Three databases of fruit images are prepared for training and validation datasets, which are improved by white noise and rotation to keep its consistency and increase distinguishability at the same time. The availability of various image descriptors and models allows to study effectiveness and efficiency of neural networks, classification accuracy, and learning process.

## Instalation

This project works under Python 3.6.8 version, which has to be forced due to deprecated interfaces of later released frameworks.

### Enviroment

1. Install Python 3.6.8 [[download]](https://www.python.org/downloads/release/python-368/) and add it to your system variables [[guide]](https://realpython.com/add-python-to-path/).
2. Install fallowing liblaries, fallow the commands in your system console
+ tensoflow v1.14
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

## Dataset

There are three different databases containing pictures of fruits:

1. Images of objects in the centre of picture with removed background.
  source: https://github.com/Horea94/Fruit-Images-Dataset/tree/3571f04df801a3c09c24905f4eccbc159dbab60c/Training

2. Pictures of fruits with complex background. Simple image has only one object in different place. Objects might be deformed due to image resizing and rotating process. 
  source: my own pictures multiplied with image editing 
  
3. Probably the hardsest problem of difficulty level. Images contains any number of fruits located randomply over the picture.
  source: http://doi.org/10.5281/zenodo.1310165
