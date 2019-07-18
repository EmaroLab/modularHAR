#!/usr/bin/env python3

import argparse

import keras
import pandas as pd
import numpy as np
import itertools

import my_modules.preprocessdata 
import my_modules.nnmodelsV4
import my_modules.offline

from my_modules.preprocessdata import Activities, IMUSensors
from my_modules.nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, loadNNModel
from my_modules.offline import ActivityModule, Classifier, Sensor, ArgMinStrategy, SingleSensorSystem, Reasoner, PyPlotter, MostFrequentStrategy, WindowSelector, SimulationResults

# # reload modules (useful for development)
# import importlib
# importlib.reload(preprocessdata)
# importlib.reload(nnmodelsV4)
# importlib.reload(offline)

# from preprocessdata import Activities, IMUSensors
# from nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, loadNNModel
# from offline import ActivityModule, Classifier, Sensor, ArgMinStrategy, SingleSensorSystem, PyPlotter, Reasoner, MostFrequentStrategy, WindowSelector

# Define the parser
parser = argparse.ArgumentParser(description='System Simulation')

# Declare an argument (`--algo`), telling that the corresponding value should be stored in the `algo` field, and using a default value if the argument isn't given
parser.add_argument('--person', action="store", dest='person', default='P10')
parser.add_argument('--session', action="store", dest='session', default='S3')
parser.add_argument('--window', action="store", dest='windowLength', default=1, type = int)
parser.add_argument('--thresh', action="store", dest='errorThreshold', default=100, type = float)
parser.add_argument('--actcat', action="store", dest='activityCategory', default='Locomotion')
parser.add_argument('--sensors', action="append", dest='sensorNames', default=[])
parser.add_argument('--activities', action="append", dest='activityNames', default=[])

# Now, parse the command line arguments and store the values in the `args` variable
args = parser.parse_args() # Individual arguments can be accessed as attributes of this object

baseDir = 'NNModels'   # NNModels base directory
#*****************************************************************************
#SYSTEM SPECIFICS
person = args.person
session = args.session

lookback = 15
sensorChannels = 6
windowLength = args.windowLength
errorThreshold = args.errorThreshold

if errorThreshold < 1:
    n = '0' + str(int(errorThreshold * 100))
else:
    n = str(int(errorThreshold)) 

simulationResultsbaseDir = f'threshold{n}_simulation_results'

# Choose activity category from BothArmsLabel, RightArmLabel, LeftArmLabel, Locomotion
activityCategory = args.activityCategory
print(f'\nActivity Catgeory: {activityCategory}')

# Choose Sensors among ['backImu' , 'llaImu', 'luaImu', 'rtImu', 'rlaImu', 'ruaImu'] 
if args.sensorNames:
    sensorNames = args.sensorNames
else:
    sensorNames = ['backImu' , 'llaImu', 'luaImu', 'rtImu', 'rlaImu', 'ruaImu']

print(f'\nSensnor Names:\n {sensorNames}')

# Choose activities among ['Walk', 'SitDown', 'StandUp', 'OpenDoor', 'CloseDoor', 'PourWater', 'DrinkGlass', 'BrushTeeth', 'CleanTable']
if args.activityNames:
    activityNames = args.activityNames
else:
    activities = Activities()
    activityNames = activities.getNamesWithCategory(activityCategory)
#activityNames = ['Walk', 'SitDown', 'StandUp']

print(f'\nActivity Names:\n {activityNames}')

classifyStrategy = ArgMinStrategy()
reasonerSelectionStartegy = MostFrequentStrategy()
windowSelectionStrategy = MostFrequentStrategy()
#*****************************************************************************
print(f'\n({person}, {session}) SYSTEM SETUP START\n')

# SYSTEM SETUP

# setup of the Classifiers and WindowSelectors  
classifiers = []
windowSelectors = []
for sensorName in sensorNames:   
    classifiers.append(Classifier(classifyStrategy, sensor = sensorName, activityCategory = activityCategory))
    windowSelectors.append(WindowSelector(windowLength, windowSelectionStrategy,  sensor = sensorName, activityCategory = activityCategory))

# load ground truth and setup of the Sensors for the plots
imuSensorsDataFrame = pd.read_csv('imuSensorsWithQuaternions.csv', header = [0,1], index_col = [0,1,2])

# setup voting module
reasoner = Reasoner(reasonerSelectionStartegy)

# load errors dataframe
errorsDf = pd.read_csv(f'{activityCategory}_errors.csv', header = [0,1], index_col = [0,1])
print(f'\n{activityCategory}_errors.csv LOADED')

print(f'\n({person}, {session}) SYSTEM SETUP COMPLETE')
#*****************************************************************************
# SIMULATION

print(f'\n({person}, {session}) SIMULATION START')

# 
tiSim = 0   # simulation initial timestep (sec = timsteps / freq)
#tfSim = 201   # simulation final  timestep (sec = timsteps / freq)
idx = pd.IndexSlice
tfSim = errorsDf.loc[idx[person,session],:].values.shape[0]

# # errors per sensor and per activity at sensor frequency
sensorSystemsErrors = np.empty((len(sensorNames), tfSim-tiSim, len(activityNames)))

# activity selected by the classifier at sensor frequency
selectedActivityId = np.empty((tfSim-tiSim, len(sensorNames)), dtype=object)   # id are dictionaries

# activity selected by window selection at sensor frequency // windowLength frequency
numOfFinalSamples = (tfSim-tiSim) // windowLength
windowSelectedActivityId = np.empty((numOfFinalSamples, len(sensorNames)), dtype=object)
windowSelectedActivityName = np.empty((numOfFinalSamples, len(sensorNames)), dtype=object) # names are string
windowResultantActivityName = np.empty(numOfFinalSamples, dtype=object)

for t in range(tiSim, tfSim):
    for i, sensorName in enumerate(sensorNames):   # for each sensor
        errorsAndIds = []   # at each time step initialize the errors
        for j, activityName in enumerate(activityNames):   # for each actvityModule in the sensorSystem
            activityError = errorsDf.loc[idx[person,session],idx[sensorName, activityName]].values[t] # get the error corresponding to current sensor and activity, at time t, from the df
            activityModuleId = {'sensor' : sensorName, 'activityCategory' : activityCategory, 'activityName' : activityName}         
            errorsAndIds.append([activityError, activityModuleId]) # append the errorAndId, format required by the classifier input 
            sensorSystemsErrors[i,t - tiSim,j] =  activityError #  store errors for plot t timestep, i sensor system, , j activity module
        
        if min([errorsAndIds[i][0] for i in range(len(errorsAndIds))]) > errorThreshold:
            selectedActivityId[t - tiSim,i] = {'sensor' : sensorName, 'activityCategory' : activityCategory, 'activityName' : 'nullActivity'}
        else:
            selectedActivityId[t - tiSim,i] = classifiers[i].classify(errorsAndIds)   # get the activityId chosen by the classifier at the curent timestep
        
        windowSelectors[i].appendId(selectedActivityId[t - tiSim,i]) # append the selected activity id to the window buffer
        if windowSelectors[i].isFull():   
            windowSelectedActivityId[(t - tiSim) // windowLength, i] = windowSelectors[i].selectIdAndClearBuffer()
            windowSelectedActivityName[(t - tiSim) // windowLength, i] = windowSelectedActivityId[(t - tiSim) // windowLength, i]['activityName']
    if (t + 1 - tiSim) % windowLength == 0:   # once every windowLength timesteps
        windowResultantActivityName[(t - tiSim) // windowLength] = reasoner.selectActivity(windowSelectedActivityName[(t - tiSim) // windowLength,:])

    if t % 100 == 0:
            print(f'simulation time {t}')

print(f'\n({person}, {session}) SIMULATION COMPLETE')
#*****************************************************************************
# instance of class to save plots and confusion matrix 
simulationResults = SimulationResults(activityCategory, person, session, windowLength, lookback, baseDir = simulationResultsbaseDir)

print(f'\n({person}, {session}) PLOT START')
# PLOT
tiPlot = tiSim # times step (30 Hz -> 30 steps per second) sec = timsteps / freq
tfPlot = tfSim # times step (30 Hz -> 30 steps per second) sec = timsteps / freq

pyplotter = PyPlotter(tiSim, tfSim, activityCategory, imuSensorsDataFrame, person, session,  windowLength = windowLength, lookback = lookback, baseDir= simulationResultsbaseDir)
for i, sensorName in enumerate(sensorNames):
    # pyplotter.plotSensorSystemErrorsV2(activityNames, 
    #                                  sensorName,
    #                                  sensorSystemsErrors[i,:,:], 
    #                                  windowSelectedActivityName[:,i], 
    #                                  tiPlot, tfPlot,
    #                                  windowLength=windowLength, 
    #                                  figsize = (20,10), top = 1, 
    #                                  toFile = True,
    #                                  majorTicks = 5, 
    #                                  minorTicks = 1 )

    pyplotter.plotSensorSystemErrorsV3(activityNames, 
                                     sensorName,
                                     sensorSystemsErrors[i,:,:], 
                                     windowSelectedActivityName[:,i], 
                                     tiPlot, tfPlot,
                                     windowLength=windowLength, 
                                     figsize = None, 
                                     top = 1, 
                                     toFile = True,
                                     majorTicks = 5, 
                                     minorTicks = 1 )

pyplotter.plotSelectedVsTrue(windowResultantActivityName, 
                             tiPlot, tfPlot, 
                             windowLength=windowLength,
                             figsize = (20, 5), 
                             top = 1, 
                             toFile = True,
                             majorTicks = 5, 
                             minorTicks = 1 )

print(f'\n({person}, {session}) PLOT END')
#*****************************************************************************
# CLASSIFICATION

print(f'\n({person}, {session}) CONFUSION MARIX EVALUATION START')

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
print(len(actualLabels))

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
    if actualLabels[i] not in activityNames:
        columnName = 'nullActivity'
    else:
        columnName = actualLabels[i]

    maxVotingConfusionMatrixDf.loc[idx[maxVotingPredictedLabels[i]], idx[columnName]] += 1

simulationResults.saveMaxVotingConfusionMatrixDf(maxVotingConfusionMatrixDf) 
# maxVotingConfusionMatrixDf.to_csv(f'max_voting_confusion_matrix.csv')


for i, sensor in enumerate(sensorNames):
    sensorPredictedLabels = [[ResultantActivityName] * windowLength for ResultantActivityName in windowSelectedActivityName[:,i]]
    sensorPredictedLabels = list(itertools.chain.from_iterable(sensorPredictedLabels))
    print(len(sensorPredictedLabels))

    # initialize i-th sensor confusion matrix
    sensorConfusionMatrixDf = pd.DataFrame(np.zeros((n,n)), columns = activityNamesWithNullActivity, index = activityNamesWithNullActivity)
    sensorConfusionMatrixDf.columns.name = 'TRUE'
    sensorConfusionMatrixDf.index.name = 'PREDICTED'

    # fill i-th sensor confusion matrix
    for i in range(len(sensorPredictedLabels)):
        if actualLabels[i] not in activityNames:
            columnName = 'nullActivity'
        else:
            columnName = actualLabels[i]

        sensorConfusionMatrixDf.loc[idx[sensorPredictedLabels[i]], idx[columnName]] += 1

    simulationResults.saveSensorConfusionMatrixDf(sensorConfusionMatrixDf, sensor)
    #sensorConfusionMatrixDf.to_csv(f'{sensor}_confusion_matrix.csv')

print(f'\n({person}, {session}) CONFUSION MARIX EVALUATION COMPLETE')








        
