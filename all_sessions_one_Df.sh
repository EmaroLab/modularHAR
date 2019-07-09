#!/usr/bin/env python3
import pandas as pd

import my_modules.atr_data_parser
from my_modules.atr_data_parser import Sensors,  Labels

people = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07','P08', 'P09', 'P10']
baseDir = './renamed_formatted_experiment_data'
peopleDict = {}
for person in people:
    sensors = Sensors(person, baseDir = baseDir)
    peopleDict[person] = sensors.getAllSensorsDfWithLabels()
    
imuSensorsWithQuaternions = pd.concat(peopleDict)
imuSensorsWithQuaternions.to_csv('imuSensorsWithQuaternions.csv')  