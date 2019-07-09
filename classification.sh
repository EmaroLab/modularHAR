#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import preprocessdata
import nndata
import nnmodelsV4
import evaluation

from preprocessdata import Activities, IMUSensors
from nndata import FileData
from nnmodelsV4 import NNModel
from evaluation import Results

# User (set sensor, activity category)
sensor = 'rlaImu'
activityCategory = 'mlBothArms'

# Grid Search (model hyperparameters)
lookback = 40
epochs = 100
modelLoss = 'mae'
scaling = 'standardized'

# Activities (get all activityNames)
activities = Activities()
activityNames = np.array(activities.getNamesWithCategory(activityCategory))
print('\nALL activity names:\n', activityNames)

# Select subsamples of activities
ind = np.array([0,2,4,5,6,7,8,9,14,15,16])
activityNames = np.array(activityNames)
activityNames = list(activityNames[ind])
print('\nSELECTED activity names:\n', activityNames)

# Set NN Models filepaths
fileData = FileData(activityCategory, sensor, lookback)
fileData.setActivityNames(activityNames)  
fileData.setScaling(scaling)
fileData.setLoss(modelLoss)
modelsFilepaths = fileData.setModelsFilepaths( verbose = True)

#Load NN Models
models =[]
for i, filepath in enumerate(modelsFilepaths):
  models.append(baseLSTMModelNoColab())
  models[i].loadFromFilepath(filepath)

#Load test samples and targets
testSamplesList, testTargetsList = fileData.loadSamplesAndTargets('test', baseDir='') 

# Classification
results = Results(models, activityNames)

n=10
results.setSamplesAndTargestLists(
     [testSamplesList[i][:n] for i in range(len(activityNames))], 
     [testTargetsList[i][:n] for i in range(len(activityNames))]
)

df_cm = results.getConfusionMatrix()
print(df_cm)

plt.figure(figsize = (15,12))
sn.heatmap(df_cm, annot=True)
plt.show()