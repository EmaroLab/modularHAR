#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

import my_modules.preprocessdata
import my_modules.nndata
import my_modules.nnmodelsV4

from my_modules.preprocessdata import Activities, IMUSensors, MultiLabelSequence, Timeseries
from my_modules.nndata import TensorNNready, SamplesAndTargetsGenerator, FileData
from my_modules.nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, saveNNModel
from my_modules.evaluation import Results

# # reload modules (useful for development)
# import importlib
# importlib.reload(preprocessdata)
# importlib.reload(nndata)
# importlib.reload(nnmodelsV4)

# from preprocessdata import Activities, IMUSensors, MultiLabelSequence, Timeseries
# from nndata import TensorNNready, SamplesAndTargetsGenerator, FileData
# from nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, saveNNModel
# from evaluation import Results

# Import dataframe with all IMU sensors and labels 
imuSensorsDataFrame = pd.read_csv('imuSensorsWithQuaternions.csv', header = [0,1], index_col = [0,1,2])

# Fill nans with zeroes
imuSensorsDataFrame = imuSensorsDataFrame.fillna(0)

# User (set sensor, activity category)
activityCategory = 'LeftArmLabel'
sensors = ['backImu', 'rlaImu', 'ruaImu', 'llaImu', 'luaImu', 'rtImu']

# Grid Search (model hyperparameters)
lookbacks = [15]

sensorChannels = 6

epochs = 500
modelLoss = 'mae'
scaling = 'standardized'

modelParams = {
    'input_shape'          : (None, sensorChannels),
    'neurons'              : 32,
    'layers'               : 1,
    'activation'           : 'relu',
    'recurrent_activation' : 'hard_sigmoid',
    'dropout'              : 0.0,
    'recurrent_dropout'    : 0.5,
    'mask_zeros'           : False,
}

fitParams = {
    'epochs' : epochs,
    'batch_size' : 128,
    'shuffle' : True,
    'validation_split' : 0.2
}

lstmModelFactory = LSTMModelFactory(modelParams)
callbacksListFactory = BaseCallbacksListFactory()

# Activities (get all activityNames)
activities = Activities()
activityNames = np.array(activities.getNamesWithCategory(activityCategory))
print('\nALL activity names:\n', activityNames)

# # Select subsamples of activities (only for 'mlBothArms' activity Category)
# ind = np.array([0,1,2]) # Locomotion
# ind = np.array([0,1,2,5,6,8]) # LeftArmLabel
# activityNames = np.array(activityNames)
#activityNames = list(activityNames[ind])

print('\nSELECTED activity names:\n', activityNames)

for sensor in sensors:
    for lookback in lookbacks: 
        # Set filepaths for keras models callbacks
        fileData = FileData(activityCategory, sensor, lookback)
        fileData.setActivityNames(activityNames)  
        fileData.setScaling(scaling)
        fileData.setLoss(modelLoss)
        modelsFilepaths = fileData.setModelsFilepaths(verbose = True)

        # IMUSensors (get single sensor dataframe)
        imuSensors = IMUSensors(imuSensorsDataFrame)
        sensorDataframe = imuSensors.singleSensorDf(sensor)

        # TrainAndTestDataGenerator (samples and targets)
        sensorGenerator = SamplesAndTargetsGenerator(sensorDataframe,  lookback = lookback, sensorChannels=sensorChannels)
        sensorGenerator.trainTestSplitDataframe()
        sensorGenerator.setActivityCategory(activityCategory)
                
        # NN Model definition and training (and svaing with callbacks)
        activityModels = []
        testSamplesList = []
        testTargetsList = []
        for i, activityName in enumerate(activityNames): 

            trainSamples, trainTargets = sensorGenerator.getTrainSamplesAndTargets(activityName, scaling=scaling)
            trainMean, trainStd = sensorGenerator.getTrainMeanAndStd(activityName)

            callbackList = callbacksListFactory.getList(modelsFilepaths[i])   # create the callbacks list which save the model at the specific filepath
            model = lstmModelFactory.getModel()  # create a new keras model at each iteration
            identifier = {
               'activityCategory' : activityCategory,
               'sensor' : sensor,
               'activityName' : activityName,
            }
            activityModels.append(NNModel(model, callbackList, identifier = identifier, lookback = lookback))
            
            #activityModels[i].setLookback(lookback)
            activityModels[i].setTrainMeanAndStd(trainMean, trainStd)
            activityModels[i].compile(modelLoss=modelLoss)
            activityModels[i].fit(trainSamples = trainSamples, trainTargets = trainTargets, fitParams = fitParams)

            saveNNModel(activityModels[i], baseDir = 'NNModels')

            testSamples, testTargets = sensorGenerator.getTestSamplesAndTargets(activityName, scaling=scaling)
            testSamplesList.append(testSamples)
            testTargetsList.append(testTargets)
        #Evaluation
        results = Results(activityModels, activityNames)
        
        results.setSamplesAndTargestLists(testSamplesList, testTargetsList)
        
        evalDf = results.getEvaluationDf()
        fileData.saveEvaluationDf(evalDf)  # Save evaluation dataframe to csv file 

        # Save evaluation heatmap to activity category folder
        # fileData.saveEvaluationHeatmap(evalDf)

        confusionMatrixDf = results.getConfusionMatrix()
        fileData.saveConfusionMatrixDf(confusionMatrixDf) # Save confusion matrix dataframe to csv file 