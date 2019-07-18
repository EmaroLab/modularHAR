#!/usr/bin/env python3

import argparse
import pandas as pd
import my_modules.offline
import my_modules.preprocessdata 
import os

from functools import reduce
from my_modules.preprocessdata import Activities
from my_modules.offline import  SimulationResults

# Define the parser
parser = argparse.ArgumentParser(description='System Simulation')

# Declare an argument (`--algo`), telling that the corresponding value should be stored in the `algo` field, and using a default value if the argument isn't given
#parser.add_argument('--person', action="store", dest='person', default='P10')
#parser.add_argument('--session', action="store", dest='session', default='S3')
parser.add_argument('--window', action="store", dest='windowLength', default=1, type = int)
parser.add_argument('--thresh', action="store", dest='errorThreshold', default=100, type = float)
parser.add_argument('--actcat', action="store", dest='activityCategory', default='Locomotion')
parser.add_argument('--sensors', action="append", dest='sensorNames', default=[])

# Now, parse the command line arguments and store the values in the `args` variable
args = parser.parse_args() # Individual arguments can be accessed as attributes of this object

testPersonsSessions = [('P01', 'S1'), ('P02', 'S2'), ('P03', 'S3'), ('P04', 'S4'), ('P05', 'S1'), ('P06', 'S4')] # final test set
evaluationPersonsSessions = [('P07', 'S1'), ('P08', 'S4')] # evaluation set
personsSessions = [('P09', 'S2'), ('P10', 'S3')] # tuning set
personsSessions = personsSessions + testPersonsSessions + evaluationPersonsSessions

# # early test
# personsSessions = [('P07', 'S1'), ('P08', 'S4')]
# print(personsSessions)

lookback = 15
sensorChannels = 6
windowLength = args.windowLength
errorThreshold = args.errorThreshold

if errorThreshold < 1:
    n = '0' + str(int(errorThreshold * 10))
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
activities = Activities()
activityNames = activities.getNamesWithCategory(activityCategory)
#activityNames = ['Walk', 'SitDown', 'StandUp']
print(f'\nActivity Names:\n {activityNames}')

maxVotingDfList = []  # list of max voting df
sensorDfListDict = {k: [] for k in sensorNames} # dictionary of list of sensors dataframes

for persSess in personsSessions:
    # instance of class to save plots and confusion matrix 
    simulationResults = SimulationResults(activityCategory, persSess[0], persSess[1], windowLength, lookback, baseDir = simulationResultsbaseDir)
    maxVotingDfList.append(simulationResults.loadMaxVotingConfusionMatrixDf())
    for sensorName in sensorNames:
        sensorDfListDict[sensorName].append(simulationResults.loadSensorConfusionMatrixDf(sensorName))

print(len(maxVotingDfList))
print(maxVotingDfList[0])

maxVotingDfSum = reduce(pd.DataFrame.add, maxVotingDfList)
# maxVotingDfSum = reduce(lambda x, y: x.add(y, fill_value=0), maxVotingDfList)
# maxVotingDfSum.reindex(index = maxVotingDfList[0].index, level = 0)

sensorDfSumDict = {k: [] for k in sensorNames}
for sensorName in sensorNames:
    sensorDfSumDict[sensorName] = reduce(pd.DataFrame.add, sensorDfListDict[sensorName])
    # sensorDfSumDict[sensorName] = reduce(lambda x, y: x.add(y, fill_value=0), sensorDfListDict[sensorName])
    # sensorDfSumDict[sensorName].reindex(index = sensorDfListDict[sensorName][0].index, level = 0)

folderPath = os.path.join(simulationResultsbaseDir, 'simulation_results', activityCategory, f'test_confusionmatrices_window{windowLength}_lb{lookback}') 
try:
    os.makedirs(folderPath)
except FileExistsError:
    # directory already exists
    pass

filepath =  os.path.join(folderPath, f"max_voting_confusion_matrix.csv")
maxVotingDfSum.to_csv(filepath)

for sensorName in sensorNames:
    filepath =  os.path.join(folderPath, f"{sensorName}_confusion_matrix.csv")
    sensorDfSumDict[sensorName].to_csv(filepath)

