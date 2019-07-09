import pandas as pd
import numpy as np
import os


class Labels(object):
  def __init__(self):
    self.lablelDict = {
        'Walk' : 1,
        'SitDown' : 2,
        'StandUp' : 3,
        'OpenDoor' : 4,
        'CloseDoor' : 5,
        'PourWater' : 6,
        'DrinkGlass' : 7,
        'BrushTeeth' : 8,
        'CleanTable' : 9,

     } 

    self.labelsDf = None
    self.labelsDfPerSessionDict = {}
    self.timestampsPerSessionDict = {} 
    
  def clean(self, labelsFilepath):
    self.labelsDf = pd.read_excel(labelsFilepath, 
                                  usecols = [1,2,4,6,8,10])
    self.removeRowsOutsideSessions()
    self.labelsStringToNumbers()
    self.labelsDf['Time [msec]'] = self.labelsDf['Time [msec]'].astype(int) 
    return self.labelsDf
    
  def removeRowsOutsideSessions(self): 
    """Remove rows outside sessions"""
  
    labelsDfList = []
    # labelsDfDict = {}
    for sessionLabel in ['S1', 'S2', 'S3', 'S4']:
      mask = self.labelsDf['SessionLabel'] == sessionLabel    
      idx = self.labelsDf.index[mask].tolist()   # find the start and end indexes of thesessionLabel an put them in a list
      self.labelsDf['SessionLabel'].loc[idx[0]:idx[1]] = sessionLabel   # rename the NaN in between the two indexes with thesessionLabel
      sessionDf = self.labelsDf.loc[idx[0]:idx[1]].iloc[1:-1].copy()   # copy a df without the start and end row
      
      self.labelsDfPerSessionDict[sessionLabel] = sessionDf   # create the corresponding dictionary
      labelsDfList.append(sessionDf)   # create a list of the session dataframes

    self.labelsDf = pd.concat(labelsDfList)
    # self.labelsDf = pd.concat(self.labelsDfPerSessionDict)
    
  def labelsStringToNumbers(self):
    """Change labels strings to corresponding labels numbers """
    
    self.labelsDf.fillna(0, inplace=True)   # nan values corresponds to null class : 0
    self.labelsDf.replace(self.lablelDict, inplace=True)

def checkEqual(iterator):
      return len(set(iterator)) <= 1

class Sensors(object):
  def __init__(self, person, baseDir = None):
    self.person = person
    self.baseDir = baseDir
   
    self.sensorWithLabelsDfList = []
    
    self.filepaths = None
    self.setFilepaths()
    
    self.sensorsMissingTimestamps = []
    
    labelsFilepath = os.path.join(self.baseDir, self.person, self.person + '.xlsx')
    labels = Labels()
    #labelsFilepath = '/content/P01/P01.xlsx'
    self.labelsDf = labels.clean(labelsFilepath)
    
  def setFilepaths(self):
    sensorNames = ['rlaImu', 'ruaImu', 'llaImu', 'luaImu', 'backImu','rtImu' ]
    self.filepaths = [os.path.join(self.baseDir, self.person, sensor + '.csv') for sensor in sensorNames]
    
  def getSensorWithLabelsDfList(self):
    for sensorFilepath in self.filepaths:
      self.sensorWithLabelsDfList.append(self.matchSensorWithLables(sensorFilepath))
    
    print(f'all sensors have the same initial timestamp: {checkEqual([df.index[0] for df in self.sensorWithLabelsDfList])}')
    print(f'all sensors have the same final timestamp: {checkEqual([df.index[-1] for df in self.sensorWithLabelsDfList])}')
    
    return self.sensorWithLabelsDfList
  
  def getAllSensorsDfWithLabels(self):
    if not self.sensorWithLabelsDfList:
       self.getSensorWithLabelsDfList()
    
    sensorWithoutLablesDfList = [df.iloc[:,:10] for df in self.sensorWithLabelsDfList]
    labels = self.sensorWithLabelsDfList[0].iloc[:,-5:]
    sensorWithoutLablesDfList.append(labels)

    allSensorsDfWithLabels = pd.concat(sensorWithoutLablesDfList, axis = 1)
    allSensorsDfWithLabels.set_index('SessionLabel',append=True, inplace = True)
    allSensorsDfWithLabels = allSensorsDfWithLabels.swaplevel(i=1, j=0, axis=0)
    self.nameColumns(allSensorsDfWithLabels)
    
    return allSensorsDfWithLabels

  def nameColumns(self, sensorDataframe):
    sensorNames = ['rlaImu', 'ruaImu', 'llaImu', 'luaImu', 'backImu','rtImu' ]
    columnNames = [[sensorNames[0]]*10 + [sensorNames[1]]*10 + [sensorNames[2]]*10 + [sensorNames[3]]*10 + [sensorNames[4]]*10 + [sensorNames[5]]*10 + ['labels']*4,
                   ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'q1', 'q2', 'q3', 'q4'] * 6 + ['BothArmsLabel', 'RightArmLabel',	'LeftArmLabel',	'Locomotion']]


    sensorDataframe.columns = columnNames

  
  def matchSensorWithLables(self, sensorFilepath):
    print(f'\nAAA {sensorFilepath}\n')
    sensorDf = self.reorderColumns(sensorFilepath)
    sensorDf = self.fillMissingTimestamps(sensorDf)
    sensorWithLabelsDf = self.matchLabelsAndSensorTimestamps(sensorDf) 
    sensorWithLabelsDf = self.removeRowsOutsideSessions(sensorWithLabelsDf)
    sensorWithLabelsDf = self.fillBeetwenLabelsStartEnd(sensorWithLabelsDf)
    return sensorWithLabelsDf
    

  def reorderColumns(self, sensorFilepath):
    colsNames = ['Time', 'q1', 'q2', 'q3', 'q4', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
    sensorDf = pd.read_csv(sensorFilepath, 
                        usecols = range(1,12),
                        names = colsNames)

    reorganizedCols = [colsNames[0]] + colsNames[5:11] + colsNames[1:5]
    sensorDf = sensorDf[reorganizedCols]
    
    return sensorDf

  def fillMissingTimestamps(self, sensorDf):
    # FILL THE MISSING TIMESTAMPS
    start = sensorDf['Time'].values[0]
    print(f'start: {start}')
    end = sensorDf['Time'].values[-1]
    print(f'end: {end}')
    #timestamps = pd.DataFrame(np.array(range(start, end + 30, 30)), columns = ['Time'])
    timestamps = np.array(range(start, end + 30, 30))
    print(f'len before : {len(sensorDf["Time"].values)}, len after : {len(timestamps)}, missing timestamps = {len(timestamps) - len(sensorDf["Time"].values)}')
    self.sensorsMissingTimestamps.append(len(timestamps) - len(sensorDf["Time"].values))
    # print(timestamps)
    sensorDf = sensorDf.set_index('Time').reindex(timestamps)
    print(f'number of filled timestamps : {sensorDf["accX"].isna().sum()}')
    
    return sensorDf

  def matchLabelsAndSensorTimestamps(self, sensorDf):
    labelsDf = self.labelsDf.copy()  
    labelsTimestamps = labelsDf['Time [msec]'].values   # n elements
    sensorTimestamps = sensorDf.index.values   # m elements
  #  print(len(labelsTimestamps))

    labelsTimstampsMatrix = np.repeat(np.array([labelsTimestamps]), len(sensorTimestamps), axis = 0)   # m x n matrix
    sensorTimstampsMatrix = np.repeat(np.array([sensorTimestamps]), len(labelsTimestamps), axis = 0)   # n x m matrix
    sensorTimstampsMatrix = sensorTimstampsMatrix.transpose()   # m x n matrix
  #  print(labelsTimstampsMatrix.shape)
  #  print(sensorTimstampsMatrix.shape)

    sensorVsLabels = np.abs(labelsTimstampsMatrix - sensorTimstampsMatrix)   # m x n matrix 
  #  print(sensorVsLabels.shape)

    indexesWhereSensorTimestampsClosestToLabelsTimestamps = np.argmin(sensorVsLabels, axis = 0)   # the values of this array are the indexes of the timestamps of the sensorDf 
                                                                                                  # that corresponds to indexes of the timestamps of the labelsDf, represented 
                                                                                                  # here by the corresponding indexes of this array 
                                                                                                  # --> index = labelsTimestamps index, value = corresponding sensorTimestamps index

    # print(indexesWhereSensorTimestampsClosestToLabelsTimestamps.shape)
    # print(indexesWhereSensorTimestampsClosestToLabelsTimestamps)

    labelsTimestamps = sensorTimestamps[indexesWhereSensorTimestampsClosestToLabelsTimestamps]   # fancy indexing, the labels timestamps are set as the correspondin senor timestamps
    # print(labelsTimestamps.shape)

    labelsDf['Time [msec]'] = labelsTimestamps
    # print(labelsDf)

    labelsDf.set_index('Time [msec]',  inplace = True)
    print(f'\nAAAAA labelsIndexLength: {len(labelsDf.index)}, sensorIndexLength: {len(sensorDf.index)}\n')
    
    sensorWithLabelsDf = pd.concat([sensorDf, labelsDf], axis = 1)
    
    return sensorWithLabelsDf

  def removeRowsOutsideSessions(self, sensorWithLabelsDf):
    sensorWithLabelsDfList = []
    # sensorWithLabelsDfDict = {}
    for sessionLabel in ['S1', 'S2', 'S3', 'S4']:
      mask =sensorWithLabelsDf['SessionLabel'] == sessionLabel
      idx =sensorWithLabelsDf.index[mask].tolist()
      sensorWithLabelsDf['SessionLabel'].loc[idx[0]:idx[-1]] = sessionLabel
      # print(sensorWithLabelsDf['SessionLabel'].loc[idx[0]:idx[-1]])

      mask = sensorWithLabelsDf['SessionLabel'] == sessionLabel
      idx = sensorWithLabelsDf.index[mask].tolist()   # find ALL the indexes corresponding to the currentsessionLabel 
      sessionDf = sensorWithLabelsDf.loc[idx]
      #  sensorWithLabelsDfDict[SessionLabel] = sessionDf   # create the corresponding dictionary
      sensorWithLabelsDfList.append(sessionDf)   # create a list of the session dataframes

    sensorWithLabelsDf = pd.concat(sensorWithLabelsDfList)
    # np.any(sensorWithLabelsDf['SessionLabel'].isna())   # check 
    
    return sensorWithLabelsDf

  def fillBeetwenLabelsStartEnd(self, sensorWithLabelsDf):
    """ Fill in between labels values for each label column 

     logic : if two consecutive labels are the same -> fill with the label 
             if two consecutive labels are not the same fill with 0  -> after filling the previous ones you can use df.fillna(0)
    """

    for activityCategoryLabel in ['BothArmsLabel', 'RightArmLabel',	'LeftArmLabel',	'Locomotion']:
      print(f'activityCategoryLabel: {activityCategoryLabel}')
      for activityLabel in range(1,10):  # ['Walk',' SitDown', 'StandUp', 'OpenDoor', 'CloseDoor', 'PourWater', 'DrinkGlass', 'BrushTeeth', 'CleanTable']:
        mask =sensorWithLabelsDf[activityCategoryLabel] == activityLabel
        idx = sensorWithLabelsDf.index[mask].tolist()
        print(f'    activityLabel: {activityLabel}, activities count = {len(idx)/2}')
        for i in range(0,len(idx) - 1,2):
          sensorWithLabelsDf[activityCategoryLabel].loc[idx[i]:idx[i+1]] =  activityLabel

      # after filling the the activities fill the remaining nan with 0 (null class)
      sensorWithLabelsDf[activityCategoryLabel].fillna(0, inplace=True)      
      sensorWithLabelsDf[activityCategoryLabel] = sensorWithLabelsDf[activityCategoryLabel].astype(int) 
        
    return sensorWithLabelsDf