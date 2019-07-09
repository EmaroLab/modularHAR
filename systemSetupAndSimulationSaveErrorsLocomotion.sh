#!/usr/bin/env python3

import keras
import pandas as pd
import numpy as np
import itertools

import my_modules.preprocessdata
import my_modules.nnmodelsV4
import my_modules.offline

from my_modules.preprocessdata import Activities, IMUSensors
from my_modules.nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, loadNNModel
from my_modules.offline import ActivityModule, Classifier, Sensor, ArgMinStrategy, SingleSensorSystem, PyPlotter, Reasoner, MostFrequentStrategy, WindowSelector

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


testPersonsSessions = [('P01', 'S1'), ('P02', 'S2'), ('P03', 'S3'), ('P04', 'S4'), ('P05', 'S1'), ('P06', 'S4')] # final test set
evaluationPersonsSessions = [('P07', 'S1'), ('P08', 'S4')] # evaluation set
personsSessions = [('P09', 'S2'), ('P10', 'S3')] # tuning set
personsSessions = personsSessions + testPersonsSessions + evaluationPersonsSessions

lookback = 15
sensorChannels = 6

# Choose activity category from BothArmsLabel, RightArmLabel, LeftArmLabel, Locomotion
activityCategory = 'Locomotion' 

print(f'\n{activityCategory}\n')

# Choose Sensors among ['backImu' , 'llaImu', 'luaImu', 'rtImu', 'rlaImu', 'ruaImu'] 
sensorNames = ['backImu' , 'llaImu', 'luaImu', 'rtImu', 'rlaImu', 'ruaImu'] 

# Activities (get all activityNames)
activities = Activities()
activityNames = activities.getNamesWithCategory(activityCategory)

print('\nSELECTED activity names:\n', activityNames)

classifyStrategy = ArgMinStrategy()
reasonerSelectionStartegy = MostFrequentStrategy()
windowSelectionStrategy = MostFrequentStrategy()
windowLength = 1

#*****************************************************************************
print('\nSESNOR SYSTEM MODULES SETUP START')

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

    print(f'\n   sensor system {sensorName} set')

print('\nSESNOR SYSTEM MODULES SETUP END')

errorsDfList = []
for person, session in personsSessions:

    print(f'\n({person}, {session}) SENSORS SETUP START')

    # setup of the Sensors
    imuSensorsDataFrame = pd.read_csv('imuSensorsWithQuaternions.csv', header = [0,1], index_col = [0,1,2])
    imuSensors = IMUSensors(imuSensorsDataFrame)
    idx = pd.IndexSlice
    identifier = {
            'activityCategory' : activityCategory,
            'sensor' : None,
            }
    sensors = []
    for sensorName in sensorNames:
        identifier['sensor'] = sensorName
        sensorDf = imuSensors.singleSensorDf(sensorName).loc[idx[person, session], :]
        sensors.append(Sensor(sensorDf, identifier = identifier, sensorChannels = sensorChannels))

        print(f'\n   sensor {sensorName} set')

    # setup aggregation Module
    reasoner = Reasoner(reasonerSelectionStartegy)

    print(f'\n({person}, {session}) SENSORS SETUP END')
    #*****************************************************************************
    # SIMULATION
    print(f'\n({person}, {session}) SYSTEM SIMULATION START\n')

    
    tiSim = 0   # simulation initial timestep (sec = timsteps / freq)
    tfSim = 10   # simulation final  timestep (sec = timsteps / freq)
    #tfSim = len(imuSensorsDataFrame.loc[idx[person, session], :].values)

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
       
        if t % 100 == 0:
            print(f'simulation time {t}')

    # t timestep, i sensor system, , j activity module
    # sensorSystemsErrors[i,:,:] rows are the timestep and columns refers to the activities
    allSensorsErrors = np.concatenate([sensorSystemsErrors[i,:,:] for i in range(len(sensorNames))], axis = 1)
    print(f'\n{sensorSystemsErrors[0,:,:].shape}, {allSensorsErrors.shape}')

    numOftimesteps = sensorSystemsErrors.shape[1]
    columnsFirstLevel = [[sensorName]*len(activityNames) for sensorName in sensorNames]
    columnsFirstLevel = list(itertools.chain.from_iterable(columnsFirstLevel))
    columns = [columnsFirstLevel, activityNames * len(sensorNames)]    
    index = [[person] * numOftimesteps, [session] * numOftimesteps]

    errorsDfList.append(pd.DataFrame(allSensorsErrors, columns = columns, index = index))

    print('\n({person}, {session}) SYSTEM SIMULATION END\n')

errorsDf = pd.concat(errorsDfList, axis = 0)
errorsDf.to_csv(f'{activityCategory}_errors.csv')










        
