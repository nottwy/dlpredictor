#Module introduction

## 1. dlBase Module
I'll write prototypes of modedls here.
### 1.1 WDNN class
### 1.2 DeepAMR class

## 2. dlTrainEvaluate Module
I'll write some cross validation methods here.

## 3. dlLog Module
I'll write some custom callbacks here.
### 3.1 CustomMCP class
inherited from keras.callbacks.ModelCheckpoint
overview: Save the best model among the epochs.
detail: The reason why I create this custom callback is I don't want my model outputs the best each time it finds one. I want it to output the best model after one train ends. Because outputting the model wastes time.

## 4. dlAdvanced Module
I'll create some subclasses of the superclasses in the module package dlBase here.