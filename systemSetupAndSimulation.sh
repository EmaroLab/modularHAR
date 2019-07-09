#!/usr/bin/env python3

import keras
import pandas as pd
import numpy as np
import itertools

import  my_modules.preprocessdata
import  my_modules.nnmodelsV4
import  my_modules.offline

from  my_modules.preprocessdata import Activities, IMUSensors
from  my_modules.nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, loadNNModel
from  my_modules.offline import ActivityModule, Classifier, Sensor, ArgMinStrategy, SingleSensorSystem, Reasoner, PyPlotter, MostFrequentStrategy, WindowSelector, SimulationResults

# # reload modules (useful for development)
# import importlib
# importlib.reload(preprocessdata)
# importlib.reload(nnmodelsV4)
# importlib.reload(offline)

# from preprocessdata import Activities, IMUSensors
# from nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, loadNNModel
# from offline import ActivityModule, Classifier, Sensor, ArgMinStrategy, SingleSensorSystem, PyPlotter, Reasoner, MostFrequentStrategy, WindowSelector

baseDir = 'NNModels'   # NNModels base directory

#*****************************************************************************
#SYSTEM SPECIFICS
person = 'P10'
session ='S3'

lookback = 15
sensorChannels = 6

activityCategory = 'Locomotion' # BothArmsLabel, RightArmLabel, LeftArmLabel, Locomotion
sensorNames = ['backImu' , 'llaImu', 'luaImu', 'rtImu', 'rlaImu', 'ruaImu'] 

# Activities (get all activityNames)
activities = Activities()
activityNames = activities.getNamesWithCategory(activityCategory)

# Select subsamples of activities (only for 'mlBothArms' activity Category)
#ind = np.array([5,6,8]) # LeftArmLabel
#ind = np.array([3,4,5,6,7,8]) # RightArmLabel 
#activityNames = np.array(activityNames)[ind]

print('\nSELECTED activity names:\n', activityNames)

classifyStrategy = ArgMinStrategy()
reasonerSelectionStartegy = MostFrequentStrategy()
windowSelectionStrategy = MostFrequentStrategy()
windowLength = 1

#*****************************************************************************
# SYSTEM SETUP
identifier = {
               'activityCategory' : activityCategory,
               'sensor' : None,
               'activityName' : None,
            }

# setup of the sensorSystems and the corresponding, windowSelections and Classifiers
sensorSystems = []
classifiers = []
windowSelectors = []
for sensorName in sensorNames:
    identifier['sensor'] = sensorName
    activityModules = []
    for activityName in activityNames:
        identifier['activityName'] = activityName
        nnModel = loadNNModel(identifier, lookback = lookback, sensorChannels = sensorChannels, baseDir = baseDir)
        activityModules.append(ActivityModule(nnModel))
    sensorSystems.append(SingleSensorSystem(activityModules))
    classifiers.append(Classifier(classifyStrategy, sensor = sensorName, activityCategory = activityCategory))
    windowSelectors.append(WindowSelector(windowLength, windowSelectionStrategy,  sensor = sensorName, activityCategory = activityCategory))


# setup of the Sensors
imuSensorsDataFrame = pd.read_csv('imuSensorsWithQuaternions.csv', header = [0,1], index_col = [0,1,2])
imuSensors = IMUSensors(imuSensorsDataFrame)
idx = pd.IndexSlice
identifier.pop('activityName')   # remove activityName from keys
sensors = []
for sensorName in sensorNames:
    identifier['sensor'] = sensorName
    sensorDf = imuSensors.singleSensorDf(sensorName).loc[idx[person, session], :]
    sensors.append(Sensor(sensorDf, identifier = identifier, sensorChannels = sensorChannels))

# setup aggregation Module
reasoner = Reasoner(reasonerSelectionStartegy)

#*****************************************************************************
# SIMULATION

# 
tiSim = 0   # simulation initial timestep (sec = timsteps / freq)
#tfSim = 100   # simulation final  timestep (sec = timsteps / freq)
tfSim = len(imuSensorsDataFrame.loc[idx[person, session], :].values)

# errors per sensor and per activity at sensor frequency
sensorSystemsErrors = np.empty((len(sensors), tfSim-tiSim, len(activityNames)))

# activity selected by the classifier at sensor frequency
selectedActivityId = np.empty((tfSim-tiSim, len(sensors)), dtype=object)   # id are dictionaries

# activity selected by window selection at sensor frequency // windowLength frequency
numOfFinalSamples = (tfSim-tiSim) // windowLength
windowSelectedActivityId = np.empty((numOfFinalSamples, len(sensors)), dtype=object)
windowSelectedActivityName = np.empty((numOfFinalSamples, len(sensors)), dtype=object) # names are string
windowResultantActivityName = np.empty(numOfFinalSamples, dtype=object)


for t in range(tiSim, tfSim):
    for i in range(len(sensors)):   # for each sensor
        sensorData = sensors[i].getDataWithTimeIndex(t)
        errorsAndIds = sensorSystems[i].getErrorsAndIds(sensorData) # errorAndIds is list of (float, dictionary)
        for j in range(len(activityNames)):   # for each actvityModule in the sensorSystem
            sensorSystemsErrors[i,t - tiSim,j] =  errorsAndIds[j][0]   #  t timestep, i sensor system, , j activity module
        selectedActivityId[t - tiSim,i] = classifiers[i].classify(errorsAndIds)   # get the activityId chosen by the classifier at the curent timestep
        windowSelectors[i].appendId(selectedActivityId[t - tiSim,i])   
        if windowSelectors[i].isFull():   
            windowSelectedActivityId[(t - tiSim) // windowLength, i] = windowSelectors[i].selectIdAndClearBuffer()
            windowSelectedActivityName[(t - tiSim) // windowLength, i] = windowSelectedActivityId[(t - tiSim) // windowLength, i]['activityName']
    if (t + 1 - tiSim) % windowLength == 0:   # once every windowLength timesteps
        windowResultantActivityName[(t - tiSim) // windowLength] = reasoner.selectActivity(windowSelectedActivityName[(t - tiSim) // windowLength,:])

#*****************************************************************************
simulationResults = SimulationResults(activityCategory, person, session, windowLength, lookback)

# PLOT
tiPlot = tiSim # times step (30 Hz -> 30 steps per second) sec = timsteps / freq
tfPlot = tfSim # times step (30 Hz -> 30 steps per second) sec = timsteps / freq

pyplotter = PyPlotter(tiSim, tfSim, activityCategory, imuSensorsDataFrame, person, session, windowLength = windowLength, lookback = lookback)
for i in range(len(sensorSystems)):
    pyplotter.plotSensorSystemErrors(sensorSystems[i], 
                                     sensorSystemsErrors[i,:,:], 
                                     windowSelectedActivityName[:,i], 
                                     tiPlot, tfPlot,
                                     windowLength=windowLength, 
                                     figsize = (20,5), top = 2, 
                                     toFile = True)

pyplotter.plotSelectedVsTrue(windowResultantActivityName, 
                             tiPlot, tfPlot, 
                             windowLength=windowLength,
                             figsize = (20,5), 
                             top = 2, 
                             toFile = True)

#*****************************************************************************
# CLASSIFICATION

activityNumToLabelDict = {
                0 : 'nullActivity',
                1 : 'Walk',
                2 : 'SitDown',
                3 : 'StandUp' ,
                4 : 'OpenDoor',
                5 : 'CloseDoor',
                6 : 'PourWater',
                7 : 'DrinkGlass',
                8 : 'BrushTeeth',
                9 : 'CleanTable',
                    }

# find the slices of the groudtruth
idx = pd.IndexSlice
actualLabelsNumIdDf = imuSensorsDataFrame.loc[idx[person, session], idx['labels', activityCategory]] # obtain the session dataframe
actualLabelsDf = actualLabelsNumIdDf.replace(activityNumToLabelDict, inplace=False)  # replace id numbers with activity labels
actualLabels = actualLabelsDf.values[tiSim:tfSim]   # select the sequence corresponding to the simulation

# labels predicted by the voting scheme
maxVotingPredictedLabels = [[ResultantActivityName] * windowLength for ResultantActivityName in windowResultantActivityName] # windowResultantActivityName has one item each windowLengthTimesteps
maxVotingPredictedLabels = list(itertools.chain.from_iterable(maxVotingPredictedLabels))

# initialize max voting confusion matrix dataframe (include nullActivity)
activityNamesWithNullActivity = activityNames + ['nullActivity']
n = len(activityNames) + 1
maxVotingConfusionMatrixDf = pd.DataFrame(np.zeros((n,n)), columns = activityNamesWithNullActivity, index = activityNamesWithNullActivity)
maxVotingConfusionMatrixDf.columns.name = 'TRUE'
maxVotingConfusionMatrixDf.index.name = 'PREDICTED'

# fill max voting confusion matrix
for i in range(len(maxVotingPredictedLabels)):
    maxVotingConfusionMatrixDf.loc[idx[maxVotingPredictedLabels[i]], idx[actualLabels[i]]] += 1

simulationResults.saveMaxVotingConfusionMatrixDf(maxVotingConfusionMatrixDf) 
# maxVotingConfusionMatrixDf.to_csv(f'max_voting_confusion_matrix.csv')


for i, sensor in enumerate(sensorNames):
    # labels predicted by i-th sensor module
    sensorPredictedLabels = windowSelectedActivityName[:,i]

    # initialize i-th sensor confusion matrix
    sensorConfusionMatrixDf = pd.DataFrame(np.zeros((n,n)), columns = activityNamesWithNullActivity, index = activityNamesWithNullActivity)
    sensorConfusionMatrixDf.columns.name = 'TRUE'
    sensorConfusionMatrixDf.index.name = 'PREDICTED'

    # fill i-th sensor confusion matrix
    for i in range(len(sensorPredictedLabels)):
        sensorConfusionMatrixDf.loc[idx[sensorPredictedLabels[i]], idx[actualLabels[i]]] += 1

    simulationResults.saveSensorConfusionMatrixDf(sensorConfusionMatrixDf, sensor)
    # sensorConfusionMatrixDf.to_csv(f'{sensor}_confusion_matrix.csv')








        
