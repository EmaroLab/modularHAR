import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import time
import os


from my_modules.preprocessdata import Activities, MultiLabelSequence, Timeseries

class TensorNNready(object):
    """a class that represents the data with the correct type and shape to train and test keras modles
    
    from a Timeseries instance, which contains a list of timeseries, create a tensor useful to train
    a neural network regresssor which uses lookback number of samples in the past to predict the next
    (dealy = 1) sample

    Attributes
    ----------
    timeseries : Timeseries object (preprocessdata module)
        a Timeseries instance that repreents timeseries of single IMU sensor and a specific activity 
    lookback : int
        the number of timesteps in the past used to predict the next (delay = 1) timestep value
    delay : int 
        the timestep in the future to predict, delay = 1 means the next timestep 
    sensorChannels : int
        number of sensors per IMU  

    Methods
    -------
    setTimeseries()
    setLookback()
    setDelay()
    samplesTargets(multivariateTimeseries, min_index=0, step=1)  
        returns samples and target tensors to train a keras model from a multivariate timeseries
    allSamplesTargets(min_index=0, step=1, scaling = None)
        returns samples and target tensor to train a keras model from all the dataframe in timeseries instance attribute 
    """
    def __init__(self, timeseries, lookback = 40, delay = 1,  sensorChannels = 6):
        self.timeseries = timeseries
        self.lookback = lookback
        self.delay = delay
        self.sensorChannels = sensorChannels

    def setTimeseries(self, timeseries):
        self.timeseries = timeseries

    def setLookback(self, lookback):
        self.lookback = lookback

    def setDelay(self, delay):
        self.delay = delay

    def samplesTargets(self, multivariateTimeseries, min_index=0, step=1):
        """returns samples and target tensors to train a keras model from a multivariate timeseries

        a sliding approach is used to select lookback number of timsteps of the multivariate timeseries,
        then a matrix with shape = (lookback, features) is obtained. These matrices are concatenated to obtain 
        a tensor in which each matrix represtens a sample
        
        Parameters
        ----------
        multivariateTimeseries : np.array (matrix)
            a multivariate timeseries where the rows are the timesteps and the columns are the features 

        Returns
        -------
        np.array 
            shape = (samples, lookback, features)
        """
        max_index = len(multivariateTimeseries) - self.delay - 1 # -1 is because indexes start at 0
        currentTimestep = min_index + self.lookback

        rows = np.arange(currentTimestep, max_index)

        samples = np.zeros((len(rows), self.lookback // step, multivariateTimeseries.shape[-1]))   # (samples, timesteps, features)
        targets = np.zeros((len(rows), multivariateTimeseries.shape[-1]))

        for j in range(len(rows)):
            indices = range(rows[j] - self.lookback, rows[j], step)   # rows starts at i (see line 12), which is at minimum i = min_index + lookback (see line 5)
            # and at each loop it is increased by the length of the batches or the remaining
            # length to the max_index (see line13)
            samples[j] = multivariateTimeseries[indices]                           # data[indices] is all the timesteps starting lookback number of steps back from current timestep j to current time step j (sampled at step frquency)
            targets[j] = multivariateTimeseries[rows[j] + self.delay]

        return samples, targets

    def allSamplesTargets(self, min_index=0, step=1, scaling = None):
        """returns samples and target tensor to train a keras model from all the dataframe in timeseries instance attribute"""
        
        if scaling == 'normalized':
            iterable = self.timeseries.normItems
        elif scaling == 'standardized':
            iterable = self.timeseries.standItems
        else:
            iterable = self.timeseries.items

        for i, self.ts in enumerate(iterable):
            self.ts = self.ts.iloc[:,:self.sensorChannels]
            if i == 0:
                samples, targets = self.samplesTargets(self.ts.values, min_index, step)
            else:
                partial_samples, partial_targets = self.samplesTargets(self.ts.values, min_index, step)
                samples = np.concatenate((samples, partial_samples), axis = 0)
                targets = np.concatenate((targets, partial_targets), axis = 0)

        return samples, targets

class SamplesAndTargetsGenerator(object):
    def __init__(self, sensorDf, lookback = 40, sensorChannels = 6):
        self.sensorDf = sensorDf
        self.lookback = lookback
        self.sensorChannels = sensorChannels

        self.activities = Activities()    
        self.delay = 1

        self.trainDf = None
        self.testDf = None

        self.trainMultiLabelSequence = None

        self.trainTimeseries = None
        self.testTimeseries = None

    def trainTestSplitDataframe(self):
        # The testing dataset is composed of data from subjects 2 and 3 (ADL4, ADL5).
        idx = pd.IndexSlice

        # leave out: final test set (P01 S1), (P02 S2), (P03 S3), (P04, S4), (P05 S1), (P06 S4);
        #            evaluation matrix test set (P07 S1), (P08 S4);
        #            system tuning test set (P09 S2), (P10 S3); 
        # remember P06 is not part of the dataset (faulty data from rt)
        trainDfList = []
        trainDfList.append(self.sensorDf.loc[idx['P01', ('S2','S3','S4')], :]) # leave out S1
        trainDfList.append(self.sensorDf.loc[idx['P02', ('S1','S3','S4')], :]) # leave out S2
        trainDfList.append(self.sensorDf.loc[idx['P03', ('S1','S2','S4')], :]) # leave out S3
        trainDfList.append(self.sensorDf.loc[idx['P04', ('S1','S2','S3')], :]) # leave out S4
        trainDfList.append(self.sensorDf.loc[idx['P05', ('S2','S3','S4')], :]) # leave out S1
        trainDfList.append(self.sensorDf.loc[idx['P06', ('S1','S2','S3')], :]) # leave out S4
        trainDfList.append(self.sensorDf.loc[idx['P07', ('S2','S3','S4')], :]) # leave out S1
        trainDfList.append(self.sensorDf.loc[idx['P08', ('S1','S2','S3')], :]) # leave out S4
        trainDfList.append(self.sensorDf.loc[idx['P09', ('S1','S3','S4')], :]) # leave out S2
        trainDfList.append(self.sensorDf.loc[idx['P10', ('S1','S2','S4')], :]) # leave out S3
        self.trainDf = pd.concat(trainDfList)

        # evaluation matrix test set (P07 S1), (P08 S4);
        testEvalMatrixDfList = []
        testEvalMatrixDfList.append(self.sensorDf.loc[idx['P07', 'S1'], :])
        testEvalMatrixDfList.append(self.sensorDf.loc[idx['P08', 'S4'], :])
        self.testDf = pd.concat(testEvalMatrixDfList)
   

    def setActivityCategory(self, activityCategory):
        self.activityCategory = activityCategory

    def getTrainTimeseries(self,activityName):
        #0. set activity catgeory
        trainMultiLabelSequence = MultiLabelSequence(self.trainDf['labels',self.activityCategory].values)

        #1.find the timestep slices corresponding to the specified activity 
        activityId = self.activities.dict[self.activityCategory][activityName]
        trainSlices = trainMultiLabelSequence.getSlicesWithLabel(activityId)

        #2. return train time series corresponding to the activity obtained by slicing the complete train dataframe
        return Timeseries(self.trainDf, trainSlices, sensorChannels = self.sensorChannels)

    def getTrainSamplesAndTargets(self, activityName, scaling = 'normalized'):
        self.trainTimeseries = self.getTrainTimeseries(activityName)

        if scaling == 'normalized':
            self.trainTimeseries.normalizeTsItems()
        elif scaling == 'standardized':
            self.trainTimeseries.standardizeTsItems()

        #3.from timeseries instance to train tensor
        trainTensorNNready = TensorNNready(self.trainTimeseries, lookback = self.lookback, delay = self.delay, sensorChannels = self.sensorChannels)
        start = time.time()
        trainSamples, trainTargets = trainTensorNNready.allSamplesTargets(scaling = scaling)
        end = time.time()
        print("Time required to compute samples and target tensors from timeseries: {0}".format(end - start))

        return trainSamples, trainTargets

    def getTrainMeanAndStd(self, activityName):
        if self.trainTimeseries is not None:
            return self.trainTimeseries.getConcatenatedDf().iloc[:,:self.sensorChannels].mean(), self.trainTimeseries.getConcatenatedDf().iloc[:,:self.sensorChannels].std()
        else:
            self.trainTimeseries = self.getTrainTimeseries(activityName)
            return self.trainTimeseries.getConcatenatedDf().iloc[:,:self.sensorChannels].mean(), self.trainTimeseries.getConcatenatedDf().iloc[:,:self.sensorChannels].std()
 
    def getTestSamplesAndTargets(self, activityName, scaling = 'normalized'):
        #0. set activity catgeory category
        testMultiLabelSequence = MultiLabelSequence(self.testDf['labels',self.activityCategory].values)

        #1.find single label sequence
        activityId = self.activities.dict[self.activityCategory][activityName]
        testSlices = testMultiLabelSequence.getSlicesWithLabel(activityId)

        #2.create time series
        self.testTimeseries = Timeseries(self.testDf, testSlices, sensorChannels = self.sensorChannels)

        #3. normalize or standardize timeseries items
        if scaling == 'normalized':
            self.testTimeseries.normalizeTsItems(maxValue = self.trainTimeseries.concatDf().iloc[:,:self.sensorChannels].max(),
                                                 minValue = self.trainTimeseries.getConcatenatedDf().iloc[:,:self.sensorChannels].min())
        elif scaling == 'standardized':
            self.testTimeseries.standardizeTsItems(meanValue=self.trainTimeseries.getConcatenatedDf().iloc[:,:self.sensorChannels].mean(),
                                                   stdValue=self.trainTimeseries.getConcatenatedDf().iloc[:,:self.sensorChannels].std())

        #4.from timeseries instance to train tensor
        testTensorNNready = TensorNNready(self.testTimeseries, lookback = self.lookback, delay = self.delay, sensorChannels = self.sensorChannels)
        start = time.time()
        testSamples, testTargets = testTensorNNready.allSamplesTargets(scaling = scaling)
        end = time.time()
        print("Time required to compute samples and target tensors from timeseries: {0}".format(end - start))

        return testSamples, testTargets

    def samplesAndTargetsAggregated(self, activityNames, scaling = 'normalized'):
        trainSamplesTargets = []
        testSamplesTargets = []
        for activityName in activityNames:
            trainSamples, trainTargets = self.getTrainSamplesAndTargets(activityName, scaling=scaling)
            testSamples, testTargets = self.getTestSamplesAndTargets(activityName, scaling=scaling)

            trainSamplesTargets.append([trainSamples, trainTargets])
            testSamplesTargets.append([testSamples, testTargets])

        return trainSamplesTargets, testSamplesTargets

class FileData(object):
    def __init__(self, activityCategory, sensor, lookback):
        self.sensor = sensor
        self.lookback = lookback
        self.activityCategory = activityCategory
        self.rootPath = f"Data/{activityCategory}/{sensor}_lb{lookback}"
                
        try:
            os.makedirs(self.rootPath)
        except FileExistsError:
            # directory already exists
            pass
    
    def setActivityNames(self, activityNames):
        self.activityNames = activityNames

    def setScaling(self, scaling):
        self.scaling = scaling
    
    def setLoss(self, loss):
        self.loss = loss

    def saveSamplesAndTargets(self, samplesTargets, testOrTrain, baseDir = ''):

        samplesTargetsFolderPath = os.path.join(baseDir, self.rootPath, testOrTrain)
        try:
            os.makedirs(samplesTargetsFolderPath)
        except FileExistsError:
            # directory already exists
            pass

        for i, activityName in enumerate(self.activityNames):
            samaplesFilename = f"{self.sensor}_{activityName}_{testOrTrain}_samples_{self.lookback}_{self.scaling}.npy"
            targetsFilename = f"{self.sensor}_{activityName}_{testOrTrain}_targets_{self.lookback}_{self.scaling}.npy"

            samplesFilepath = os.path.join(samplesTargetsFolderPath, samaplesFilename)
            targetsFilepath = os.path.join(samplesTargetsFolderPath, targetsFilename)

            np.save(samplesFilepath, samplesTargets[i][0])
            np.save(targetsFilepath, samplesTargets[i][1])

    def loadSamplesAndTargets(self, testOrTrain, baseDir=''):
        """ Returns two lists: a list of test samples tensors and a list of test targets tensors from files in the base directory

        each samples tensor and each targets tensor in the list corresponds to the activity in activityNames
        """

        samplesTargetsFolderPath = os.path.join(baseDir, self.rootPath, testOrTrain)
        samplesTargetsFilepaths = []
        for activityName in self.activityNames:
            samaplesFilename = f"{self.sensor}_{activityName}_{testOrTrain}_samples_{self.lookback}_{self.scaling}.npy"
            targetsFilename = f"{self.sensor}_{activityName}_{testOrTrain}_targets_{self.lookback}_{self.scaling}.npy"

            samplesFilepath = os.path.join(samplesTargetsFolderPath, samaplesFilename)
            targetsFilepath = os.path.join(samplesTargetsFolderPath, targetsFilename)
            samplesTargetsFilepaths.append((samplesFilepath, targetsFilepath))

        samples = []
        targets = []
        for filepaths in samplesTargetsFilepaths:
            samples.append(np.load(filepaths[0]))
            targets.append(np.load(filepaths[1]))

        return samples, targets

    def setModelsFilepaths(self, baseDir = '', verbose = True):
        # Set NN Models filepaths
        
        modelsFolderPath = os.path.join(baseDir, self.rootPath, "models")
        try:
            os.makedirs(modelsFolderPath)
        except FileExistsError:
            # directory already exists
            pass

        modelDescription = f"{self.sensor}_lb{self.lookback}_loss{self.loss}_{self.scaling}"
        
        modelsFilepaths =  [os.path.join(modelsFolderPath, f"model_{self.activityCategory}_{activityName}_{modelDescription}.h5")
                             for activityName in self.activityNames]
        if verbose:
            print('\nmodels filepaths:\n', modelsFilepaths)
        
        return modelsFilepaths
    
    def saveEvaluationDf(self, dataframe,baseDir = ''):
        # Save to sensor filepath
        evalDfSensorFolderPath = os.path.join(baseDir, self.rootPath, "evaluation")
        try:
            os.makedirs(evalDfSensorFolderPath)
        except FileExistsError:
            # directory already exists
            pass

        evalDfDescription = f"{self.sensor}_lb{self.lookback}_loss{self.loss}_{self.scaling}"
        evalDfSensorFilepath =  os.path.join(evalDfSensorFolderPath, f"evalDf__{evalDfDescription}.csv")

        dataframe.to_csv(evalDfSensorFilepath)  

        # save to activity category file path for easier comparison
        evalDfActivityCategoryFolderPath = os.path.join(baseDir, f"Data/{self.activityCategory}", "evaluation")
        try:
            os.makedirs(evalDfActivityCategoryFolderPath)
        except FileExistsError:
            # directory already exists
            pass
        
        evalDfActivityCategoryFilepath =  os.path.join(evalDfActivityCategoryFolderPath, f"evalDf__{evalDfDescription}.csv")
        dataframe.to_csv(evalDfActivityCategoryFilepath) 
    
    def saveConfusionMatrixDf(self, dataframe, baseDir = ''):
        confusionMatrixDfFolderPath = os.path.join(baseDir, self.rootPath, "confusionMatrix")
        try:
            os.makedirs(confusionMatrixDfFolderPath)
        except FileExistsError:
            # directory already exists
            pass
        
        confusionMatrixDfDescription = f"{self.sensor}_lb{self.lookback}_loss{self.loss}_{self.scaling}"
        confusionMatrixDfSensorFilepath =  os.path.join(confusionMatrixDfFolderPath, f"confusionMatrixDf__{confusionMatrixDfDescription}.csv")
        
        dataframe.to_csv( confusionMatrixDfSensorFilepath)  

        # save to activity category file path for easier comparison
        confusionMatrixDfActivityCategoryFolderPath = os.path.join(baseDir, f"Data/{self.activityCategory}", "confusionMatrix")
        try:
            os.makedirs(confusionMatrixDfActivityCategoryFolderPath)
        except FileExistsError:
            # directory already exists
            pass
        
        confusionMatrixDfActivityCategoryFilepath =  os.path.join(confusionMatrixDfActivityCategoryFolderPath, f"confusionMatrixDf__{confusionMatrixDfDescription}.csv")
        dataframe.to_csv(confusionMatrixDfActivityCategoryFilepath) 


    def saveEvaluationHeatmap(self, dataframe,baseDir = ''):
        evalHeatmapFolderPath = os.path.join(baseDir, f"Data/{self.activityCategory}", "evaluation")
        try:
            os.makedirs(evalHeatmapFolderPath)
        except FileExistsError:
            # directory already exists
            pass

        evalHeatmapDescription = f"{self.sensor}_lb{self.lookback}_loss{self.loss}_{self.scaling}"
        evalHeatmapFilepath =  os.path.join(evalHeatmapFolderPath, f"evalHeatmap__{evalHeatmapDescription}.png")

        fig = plt.figure(figsize = (15,12))
        sns.heatmap(dataframe, annot=True)
        fig.savefig(evalHeatmapFilepath)


