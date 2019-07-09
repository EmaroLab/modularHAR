import pandas as pd
import numpy as np
from numpy_ringbuffer import RingBuffer
from abc import ABC
import os

import matplotlib
matplotlib.use('Agg')   # uncomment if do not have a graphic session
import matplotlib.pyplot as plt

from my_modules.preprocessdata import Activities, IMUSensors, MultiLabelSequence


class Sensor(object):
    """ a class to represent a single IMU sensor 
    
    Parameters
    ----------
    sensorDf : pandas dataframe
        the dataframe that contains the multivariate timeseiries 
    
    Methods
    -------
    getDataWithTimeIndex(t)
        returns the vector of sensor values corresponding to the t-th timestep
    """

    def __init__(self, sensorDatarame, identifier = None, sensorChannels = 6):
        self.sensorDf = sensorDatarame
        self.sensorChannels = sensorChannels
        if identifier is not None:
            self.identifier = {
                    'activityCategory' : identifier['activityCategory'],
                    'sensor' : identifier['sensor'],
                }

    def getDataWithTimeIndex(self, t):
        """returns the vector of sensor values corresponding to the t-th timestep
        
        Parameters
        ----------
        t : int 
            timstep index
        
        Returns
        -------
        np.array (vector)
            vector contining the sensor values
        """

        return self.sensorDf.iloc[t,:self.sensorChannels].values

class Buffer(object):
    """ a class to represent a circular buffer

    Parameters
    ----------
    ringBuffer : numpy_ringbuffer RingBuffer object
        the circular buffer
    windowLength : int
        the capacity of the buffer
    sensorChannels : int
        the number of features of the multivariate timeseries whose
        timestep are stored in the buffer

    Methods
    -------
    append(sensorData) 
        adds a numpy array of shape = (sensorChannels,) to the buffer after its last element 
    getBufferedData()
        returns all the elements in the buffer as a numpy array
    """ 

    def __init__(self, windowLength, sensorChannels):
        self.ringBuffer = RingBuffer(capacity=windowLength, dtype=(float, sensorChannels))
        self.sensorChannels = sensorChannels
        self.windowLength = windowLength    
    
    def append(self, sensorData):
        """adds a numpy array of shape = (sensorChannels,) to the buffer after its last element"""
        
        self.ringBuffer.append(sensorData)
    
    def getBufferedData(self):
        """returns all the elements in the buffer as a numpy array

        when the buffer is full the numpy array which is returned has 
        shape = (windowLength, sensorChannels) otherwise shape[0] of the
        returned numpy array is equal to the number of elements in the buffer
        """
        if not self.ringBuffer:  # first time when buffer is empty
            return np.zeros((1, self.windowLength, self.sensorChannels)) 
        return np.array(self.ringBuffer)

class Standardizer(object):
    """a class to standardize numpy arrays
    
    Parameters 
    ----------
    mean : numpy array
        each element of the array is the mean of of all timesteps of a feature
        of the all the multivariate timeseries that constitute the training data relative to a
        single activity
    std : numpy array
        each element of the array is the standard deviation of of all timesteps of a feature
        of the all the multivariate timeseries that constitute the training data relative to a
        single activity 
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def standardize(self, inputData):
        """ returns the input standardized

        Parameters
        ----------
        inputData : np.array
            the vector to be standardized which must have the same shape as mean and std attribute
        
        Returns 
        -------
        np.array
        """

        return (inputData - self.mean) / self.std

class NNDataAdapter(object):
    """ a class to adapt sensor data numpy array so that can be fed to the keras model
    
    samples of the keras models have shape = (None, None, sensorChannels) while targets have shape (None,sensorChannels),
    this class takes buffered sensor data with shape (None, sensorChannels) or sensor data with shape (sensorChannels,)
    and gives them the correct shape
    """

    def __init__(self):
        pass
    
    def adapt(self, sensorData):
        if len(sensorData.shape) == 2 or len(sensorData.shape) == 1:
            sensorData = sensorData[None,:]
        return sensorData

class ErrorEvaluator(object):
    """ a class that reprsents the block responsible to compute the prediction error

    Methods
    -------
    mae(yPred, yTrue)
        returns the mean absolute error (uses the same formulation as keras)
    mape(yPred, yTrue)
        returns the mean absolute percentage error (use the same formulation as keras)
        https://github.com/keras-team/keras/blob/master/keras/losses.py
    """
    def __init__(self, errorStartegy):
        self.errorStartegy = errorStartegy

    # def mae(self, yPred, yTrue):
    #     return np.mean(abs(yPred - yTrue))  
    
    # def mape(self, yPred, yTrue):
    #     diff = abs((yTrue - yPred) / abs(yPred))
    #     return 100. * np.mean(diff)
    
    def evaluate(self, yPred, yTrue):
        return self.errorStartegy.evaluate(yPred, yTrue)

class MaeStrategy(object):
    def evaluate(self, yPred, yTrue):
        return np.mean(abs(yPred - yTrue))  

class MapeStrategy(object):
    def evaluate(self, yPred, yTrue):
        diff = abs((yTrue - yPred) / abs(yPred))
        return 100. * np.mean(diff)

class ActivityModule(object):
    """a class that represents the basic module of the classification system
    
    it takes in the sensor data at time t and outputs the prediction error.
    it is composed of a standardizer, a buffer, an adapter and a regressor model 

    Atributes
    ---------
    identifier : dictionary
        keys are 'sensor', 'activityCategory', 'activityName'
    nnModel : NNModel object (nnmodelsV4 module)
        wrapper for the neural network regressor keras model
    buffer : Buffer object (offline module)
    
    standardizer : Standardizer object (offline module)
    
    nnDataAdapter : NNDataAdapter object (offline module)

    Methods
    -------
    setId(activityCategory, sensor, activityName)

    getPrediction(sensorData, scaling = 'standardized')
        returns the prediction given sensorData
    getErrorAndId(sensorData, scaling = 'standardized')
        implements the behaviour of the activiy module and returns the prediction 
        error and the id attribute 
    """
 
    def __init__(self, nnModel):
        """K=
        Parameters
        ----------
        nnModel : NNModel object (nnmodlesV4 module)
            the neural network module responsible for making the predictions
        sensorChannels : int, optional
            the number of features of the neural network
        """

        self.identifier = nnModel.identifier
        self.nnModel = nnModel
        self.buffer = Buffer(nnModel.lookback, nnModel.sensorChannels)
        self.standardizer = Standardizer(nnModel.trainMean, nnModel.trainStd)
        self.nnDataAdapter = NNDataAdapter()
        errorStrategy = MaeStrategy()
        self.errorEvaluator = ErrorEvaluator(errorStrategy)
    
    def getPrediction(self, sensorData, scaling = 'standardized'):
        """ returns the prediction given sensorData

        the sensor data first is standardized (optional), then it is added to the 
        buffer. At this point the buffered data neural network regressor to compute 
        the prediction

        Parameters
        ----------
        sensorData : np.array (1 dimensional)
            data coming from the sesnor at a time certain time instant
        scaling : str, optional
            set the scaling according to the type how data the neural network 
            model was traind on (standardized, normalized or None)
        
        Returns
        -------
        numpy array 
            shape = (1,sensorChannels)
        """

        if scaling == 'standardized':
            sensorData = self.standardizer.standardize(sensorData)
        # first get the buffered sensor data up to the previous time step and adapt for the nn
        bufferedData = self.buffer.getBufferedData()
        nnAdaptedBufferedData = self.nnDataAdapter.adapt(bufferedData)

        # then append the sensor data of the current time step t
        self.buffer.append(sensorData) 

        # use the buffered sensor data up to the previous time step (t-1) to make the prediction for the current time step P(t)
        prediction = self.nnModel.predict(nnAdaptedBufferedData)
        return prediction[0,:]   # prediction[0,:] from shape (1,sensorChannels) to (sensorChannels,) (adapter)
        
    def getErrorAndId(self, sensorData, scaling = 'standardized'):
        """ returns the prediction error and the id attribute given the sensorData

        the sensor data first is standardized (optional), then it is added to the 
        buffer. At this point the buffered data neural network regressor to compute 
        the prediction. In the end the prediction and the actual vlalues are used to 
        compute the prediction error

        Parameters
        ----------
        sensorData : np.array (1 dimensional)
            data coming from the sesnor at a time certain time instant
        scaling : str, optional
            set the scaling according to the type how data the neural network 
            model was traind on (standardized, normalized or None)
        
        Returns
        -------
        float
            the prediction error
        dictionary
            the identifier attribute
        """

        if scaling == 'standardized':
            sensorData = self.standardizer.standardize(sensorData)
         # first get the buffered sensor data up to the previous time step and adapt for the nn
        bufferedData = self.buffer.getBufferedData()
        nnAdaptedBufferedData = self.nnDataAdapter.adapt(bufferedData)

        # then append the sensor data of the current time step t
        self.buffer.append(sensorData) 

        # adaptedSensorData = self.nnDataAdapter.adapt(sensorData)
        # error = self.nnModel.evaluate(nnAdaptedBufferedData, adaptedSensorData)

        # use the buffered sensor data up to the previous time step (t-1) to make the prediction for the current time step P(t)
        prediction = self.nnModel.predict(nnAdaptedBufferedData)

        # compute the error between P(t) and A(t)
        error = self.errorEvaluator.evaluate(prediction[0,:], sensorData)   # prediction[0,:] from shape (1,6) to (6,) (adapter)
        return error, self.identifier

class SingleSensorSystem(object):
    """ a class that represent a collection of activity modules"""
    def __init__(self, activityModules):
        """ 
        Parameters 
        ----------
        activityModules : list of activityModule objects

        Methods
        -------
        getErrorsAndIds(sensorData)
            returns thr list of errors and corresponding ids of the modules in the system
        """
        self.activityModules = activityModules
        self.identifier = {
               'activityCategory' : activityModules[0].identifier['activityCategory'],
               'sensor' : activityModules[0].identifier['sensor'],
            }
        
    def getErrorsAndIds(self, sensorData):
        """ returns thr list of errors and corresponding ids of the modules in the system
        
        Parameters
        ----------
        sensorData : np.array (1 dimensional)
            data coming from the sesnor at a time certain time instant shape = (sensorChannels,) 
        
        Returns
        -------
        list of (float, dictionary)
            the float is the error and the dictionary is the identifier of the corresponding 
            activity module, its keys are 'sensor', 'activityCategory', 'activityName'
        """

        errorsAndIds = []
        for activityModule in self.activityModules:
            errorsAndIds.append(activityModule.getErrorAndId(sensorData))
        return errorsAndIds

class Classifier(object):
    """ a class to classify activities using predictions errors from activity modules
    
    Methods
    -------
    classify(*args)
        returns the id of the activity module output with lowest prediction error 
        among all activity modules
    """
    def __init__(self, classifyStrategy, sensor = None, activityCategory = None):
        self.classifyStrategy = classifyStrategy
        self.identifier = {
            'activityCategory' : activityCategory,
            'sensor' : sensor,
        }
    
    def classify(self, errorsAndIds):
        """ returns the id of the activity module outputs that has been chosen by the strategy
        
        Parameters
        ----------
        args : (error, activityId) 
            output of the activity module (what the method getErrorAndId returns) 
            error is a float, activityId is a dictionary
        
        Returns
        -------
        dictionary
            activityId of the activity module with lowest error. the dictionary keys 
            are  'sensor', 'activityCategory', 'activityName'
        """
        return self.classifyStrategy.classify(errorsAndIds)
        
class ArgMinStrategy(object):
    def classify(self, errorsAndIds):
        """ returns the id of the activity module outputs with lowest prediction error
        
        returns the the second element of arg which has lowest second first among all args
        args are tuples (error, activityId), which are the outpus of the activity modules

        Parameters
        ----------
        args : (error, activityId) 
            output of the activity module (what the method getErrorAndId returns) 
            error is a float, activityId is a dictionary
        
        Returns
        -------
        dictionary
            activityId of the activity module with lowest error
        """

        errors = np.array([errorAndId[0] for errorAndId in errorsAndIds])
        activityId = errorsAndIds[np.argmin(errors)][1]  # fancy indexing
        return activityId

class WindowSelector(object):
    """ a class that represnt the module that chooses and id over a temporal sequence of ids
    

    Attributes
    ----------
    windowLength : int 
        number of activity identifiers in the sequence
    selectionStrategy : a startegy object to select the ids
   
    buffer : list of dictionary
        each dictionary is an activity identifier, its keys 
        are 'sensor', 'activityCategory', 'activityName'
    
    Methods
    -------
    isFull()
        return true if the number of id in the list is equal to windowLength attribute
    
    appendId(identifier)
        appends the identifier to the buffer attribute if it is not full
    
    selctIdAndClearBuffer()
        uses the sectionStrtategy attribute to select the id frm those in the buffer
        and clear the buffer

    """
    def __init__(self, windowLength, selectionStrategy,  sensor = None, activityCategory = None):
        self.windowLength = windowLength
        self.selectionStartegy = selectionStrategy
        self.buffer = []

        self.identifier = {
            'activityCategory' : activityCategory,
            'sensor' : sensor,
        }
        
    def isFull(self):
        return len(self.buffer) == self.windowLength
    
    def appendId(self, identifier):
        if not self.isFull():
            self.buffer.append(identifier)
    
    def selectActivity(self, activityNames):
        """ returns activityName from a list of activityNames based on the startegy attribute
        
        Parameters
        ----------
        activityNames : list of str

        Returns
        -------
        str
            the activityName
        """
        
        return self.selectionStartegy.selectActivity(activityNames)

    def selectIdAndClearBuffer(self):
        """returns the id whose activity name is chosen by the selectionStartegy atrribute selectActivity method

        and clears the buffer attribute

        Returns
        dictionary
            whose activity name is chosen by the selectionStartegy atrribute selectActivity method
            the dictionary keys 
            are  'sensor', 'activityCategory', 'activityName'
        """
        activityNames = [identifier['activityName'] for identifier in self.buffer]
        selectedActivityName = self.selectActivity(activityNames)
        selectedId = self.identifier
        selectedId['activityName'] = selectedActivityName
        self.buffer = []
        
        return selectedId

class Reasoner(object):
    """ a clas to represent the module that aggregates informations from the sensors systems"""
    def __init__(self, selectionStrategy): 
        self.selectionStrategy = selectionStrategy
    
    def selectActivity(self, activities):
        return self.selectionStrategy.selectActivity(activities)
        
class MostFrequentStrategy(object):
    def selectActivity(self, activities):
        """returns the most frequent activity in the activities list
        
        Parameters
        ----------
        activities : list of str
            each the str is an activityName
        """
        activitiesList = list(activities)   # in case it is a np.array
        return max(set(activitiesList), key = activitiesList.count)   # when mutiple activities have the same count retuns one (not specified)  -> address the problem
  
class ProbabilityEvaluator(object):
    """ computes [(1/err_i) / sum_i_n(1/err_i) of i from 0 to n]
    Parameters
        ----------
        args : (error, activityId) 
            output of the activity module (what the method getErrorAndId returns) 
            error is a float, activityId is a dictionary
        
        Returns
        -------
        dictionary
            activityId of the activity module with lowest error
        """

    def evaluate(self, *args):
        invertedErrorSum = sum([1/arg[0] for arg in args])
        return [((1/arg[0]) / invertedErrorSum, arg[1]) for arg in args]
    
class PyPlotter(object):
    """a class that represents the scope of he simulation"""
    def __init__(self, tiSim, tfSim, activityCategory, imuSensorsDataFrame, person, session, windowLength = 1, lookback = 15, groundTruthFreq = 30):
        self.tiSim = tiSim
        self.tfSim = tfSim
        self.activityCategory = activityCategory
        self.imuSensorsDataFrame = imuSensorsDataFrame
        self.imuSensors = IMUSensors(imuSensorsDataFrame)
        self.activities = Activities()
        self.groundTruthFreq = groundTruthFreq
        self.person = person
        self.session = session
        self.simulationResults = SimulationResults(activityCategory, person, session, windowLength, lookback)
        self.colorDict = {
            'BothArmsLabel' : {
             0       : 'k',             # 'nullActivity'   
             1   : 'g',          # 'Walk'    
             2 : '#FC33FF',            # 'SitDown'   
             3   : 'b',          # 'StandUp' 
             4 :  '#33FFE3',            # 'OpenDoor'    
             5    : 'r',         # 'CloseDoor'      
             6    : 'y',         # 'PourWater'        
             7    : 'm',         # 'DrinkGlass'       
             8    : 'c',         # 'BruskTeeth'  
             9   : '#581845',    # 'CleanTable'       
            },

            'RightArmLabel' : {
             0       : 'k',             # 'nullActivity'   
             1   : 'g',          # 'Walk'    
             2 : '#FC33FF',            # 'SitDown' 
             3   : 'b',          # 'StandUp' 
             4 :  '#33FFE3',            # 'OpenDoor' 
             5    : 'r',         # 'CloseDoor'      
             6    : 'y',         # 'PourWater'        
             7    : 'm',         # 'DrinkGlass'       
             8    : 'c',         # 'BruskTeeth'  
             9   : '#581845',    # 'CleanTable'       
            },

            'LeftArmLabel' : {
             0       : 'k',             # 'nullActivity'   
             1   : 'g',          # 'Walk'    
             2 : '#FC33FF',            # 'SitDown'     
             3   : 'b',          # 'StandUp' 
             4 :  '#33FFE3',            # 'OpenDoor'  
             5    : 'r',         # 'CloseDoor'      
             6    : 'y',         # 'PourWater'        
             7    : 'm',         # 'DrinkGlass'       
             8    : 'c',         # 'BruskTeeth'  
             9   : '#581845',    # 'CleanTable'       
            },

            'Locomotion' : {
             0       : 'k',             # 'nullActivity'   
             1   : 'g',          # 'Walk'    
             2 : '#FC33FF',            # 'SitDown' (nullActivity)     
             3   : 'b',          # 'StandUp' 
            },
        }
    
    def plotSensorSystemErrors(self,  sensorSystem, sensorSystemErrors, selectedActivityName, tiPlot, tfPlot, figsize = (400,10), top = 2, toFile = False, windowLength = 1):
        """ plot the errors, the ground truth and the predicted labels of a sensor system
        
        Parameters
        ----------
        sensorSystem : SingleSensorSystem (offline module object)

        sensorSystemErrors : numpay array (shape = (timesteps, num of ActivityModules))
            the errors timsteps of all the activity modules in the sensor system 
        selectedActivityName : list of str
            the list of the activities selected by the classifier
        tiPlot: int 
            the intial time step of the plot
        tfPlot : int
            the final time step of the plot
        top : int, optional 
            the upper limit of the y axis
        toFile : bool
            if true the plot instead of being shown is saved to file
        """
        predfreq = self.groundTruthFreq / windowLength

        activityNames = [activityModule.identifier['activityName'] for activityModule in sensorSystem.activityModules]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        # plot the errors of all the activity modules that compose the sensor system
        for i in range(sensorSystemErrors.shape[-1]):
            activityId = self.activities.dict[self.activityCategory][activityNames[i]]
            plt.plot(np.array(range(tiPlot, tfPlot))/self.groundTruthFreq,   #   sensor errors timesteps are at sensor frequency 
                     sensorSystemErrors[tiPlot-self.tiSim:tfPlot-self.tiSim, i], 
                     self.colorDict[self.activityCategory][activityId], label=activityNames[i])
        
        # find the slices of the groudtruth (groundtruth is at sensor frequency)
        idx = pd.IndexSlice
        trueLablesSequence = self.imuSensorsDataFrame.loc[idx[self.person, self.session, :], idx['labels', self.activityCategory]].values[tiPlot:tfPlot]
        trueMultiLabelSequence = MultiLabelSequence(trueLablesSequence)
        trueSlices, trueLabels = trueMultiLabelSequence.getSlicesAndLabelsLists()

        # find the slices of the predicted labels
        activityNumId = [self.activities.dict[self.activityCategory][activityName] for activityName in selectedActivityName]
        
        # the activity id is saved in list (frequency = sensorfrequency / windowlength) starting from tiSim timestep, 
        # here select subsequences starting from the index correspondin to tiPlot timstep
        predictedMultiLabelSequence = MultiLabelSequence(activityNumId[(tiPlot-self.tiSim)//windowLength:(tfPlot-self.tiSim)//windowLength])     
        predictedSlices, predictedLabels = predictedMultiLabelSequence.getSlicesAndLabelsLists()

        # plot the groundtruth labels as background colors in the top half of the plot 
        for i, item in enumerate(trueSlices):
            plt.axvspan((item.start + tiPlot)/self.groundTruthFreq, (item.stop + tiPlot)/self.groundTruthFreq,   # divide by the sensor frequency to obtain the time in seconds                          
                        facecolor=self.colorDict[self.activityCategory][trueLabels[i]], 
                        #alpha=0.3, 
                        ymin=0, ymax=0.5)

        # plot the predicted labels as background colors in the bottom half of the plot 
        for i, item in enumerate(predictedSlices):
            # divide tiPlot by the sensor frequency to obtain the time in seconds at which the first slice starts, the predicted labels are saved
            # at predfrequency = sensorfrequency / windowlength  
            plt.axvspan((item.start/predfreq + tiPlot/self.groundTruthFreq), (item.stop/predfreq + tiPlot/self.groundTruthFreq),                            
                        facecolor=self.colorDict[self.activityCategory][predictedLabels[i]], 
                        #alpha=0.3, 
                        ymin=0.5, ymax=1)

        plt.text(tiPlot/self.groundTruthFreq, 0.75*top, r'PREDICTED', fontsize = 20)
        plt.text(tiPlot/self.groundTruthFreq, 0.25*top, r'TRUE', fontsize = 20)  
        
        major_ticks = np.arange(tiPlot/self.groundTruthFreq, tfPlot/self.groundTruthFreq, 1)
        minor_ticks = np.arange(tiPlot/self.groundTruthFreq, tfPlot/self.groundTruthFreq, 0.5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        plt.grid(b=True, which='both', axis='x')

        plt.xlim(left=tiPlot/self.groundTruthFreq)
        plt.ylim(top=top, bottom = 0)

        plt.xlabel('seconds')

        plt.legend(loc = 'upper left')

        if toFile:
            self.simulationResults.saveSensorPlotImage(plt, sensorSystem.identifier['sensor'])
            #plt.savefig(f"{sensorSystem.identifier['sensor']}_errorplot.png", bbox_inches ='tight')
            #plt.close(fig) 
        else:
            plt.show()

    
    def plotSensorSystemErrorsV2(self,  activityNames, sensorName, sensorSystemErrors, selectedActivityName, tiPlot, tfPlot, figsize = (400,10), top = 2, toFile = False, windowLength = 1):
        """ plot the errors, the ground truth and the predicted labels of a sensor system
        
        Parameters
        ----------
        sensorSystem : SingleSensorSystem (offline module object)

        sensorSystemErrors : numpay array (shape = (timesteps, num of ActivityModules))
            the errors timsteps of all the activity modules in the sensor system 
        selectedActivityName : list of str
            the list of the activities selected by the classifier
        tiPlot: int 
            the intial time step of the plot
        tfPlot : int
            the final time step of the plot
        top : int, optional 
            the upper limit of the y axis
        toFile : bool
            if true the plot instead of being shown is saved to file
        """
        predfreq = self.groundTruthFreq / windowLength

        #activityNames = [activityModule.identifier['activityName'] for activityModule in sensorSystem.activityModules]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        # plot the errors of all the activity modules that compose the sensor system
        for i in range(sensorSystemErrors.shape[-1]):
            activityId = self.activities.dict[self.activityCategory][activityNames[i]]
            plt.plot(np.array(range(tiPlot, tfPlot))/self.groundTruthFreq,   #   sensor errors timesteps are at sensor frequency 
                     sensorSystemErrors[tiPlot-self.tiSim:tfPlot-self.tiSim, i], 
                     self.colorDict[self.activityCategory][activityId], label=activityNames[i])
        
        # find the slices of the groudtruth (groundtruth is at sensor frequency)
        idx = pd.IndexSlice
        trueLablesSequence = self.imuSensorsDataFrame.loc[idx[self.person, self.session, :], idx['labels', self.activityCategory]].values[tiPlot:tfPlot]
        trueMultiLabelSequence = MultiLabelSequence(trueLablesSequence)
        trueSlices, trueLabels = trueMultiLabelSequence.getSlicesAndLabelsLists()

        # find the slices of the predicted labels
        activityNumId = [self.activities.dict[self.activityCategory][activityName] for activityName in selectedActivityName]
        
        # the activity id is saved in list (frequency = sensorfrequency / windowlength) starting from tiSim timestep, 
        # here select subsequences starting from the index correspondin to tiPlot timstep
        predictedMultiLabelSequence = MultiLabelSequence(activityNumId[(tiPlot-self.tiSim)//windowLength:(tfPlot-self.tiSim)//windowLength])     
        predictedSlices, predictedLabels = predictedMultiLabelSequence.getSlicesAndLabelsLists()

        # plot the groundtruth labels as background colors in the top half of the plot 
        for i, item in enumerate(trueSlices):
            plt.axvspan((item.start + tiPlot)/self.groundTruthFreq, (item.stop + tiPlot)/self.groundTruthFreq,   # divide by the sensor frequency to obtain the time in seconds                          
                        facecolor=self.colorDict[self.activityCategory][trueLabels[i]], 
                        #alpha=0.3, 
                        ymin=0, ymax=0.5)

        # plot the predicted labels as background colors in the bottom half of the plot 
        for i, item in enumerate(predictedSlices):
            # divide tiPlot by the sensor frequency to obtain the time in seconds at which the first slice starts, the predicted labels are saved
            # at predfrequency = sensorfrequency / windowlength  
            plt.axvspan((item.start/predfreq + tiPlot/self.groundTruthFreq), (item.stop/predfreq + tiPlot/self.groundTruthFreq),                            
                        facecolor=self.colorDict[self.activityCategory][predictedLabels[i]], 
                        #alpha=0.3, 
                        ymin=0.5, ymax=1)

        plt.text(tiPlot/self.groundTruthFreq, 0.75*top, r'PREDICTED', fontsize = 20)
        plt.text(tiPlot/self.groundTruthFreq, 0.25*top, r'TRUE', fontsize = 20)  
        
        major_ticks = np.arange(tiPlot/self.groundTruthFreq, tfPlot/self.groundTruthFreq, 1)
        minor_ticks = np.arange(tiPlot/self.groundTruthFreq, tfPlot/self.groundTruthFreq, 0.5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        plt.grid(b=True, which='both', axis='x')

        plt.xlim(left=tiPlot/self.groundTruthFreq)
        plt.ylim(top=top, bottom = 0)

        plt.xlabel('seconds')

        plt.legend(loc = 'upper left')

        if toFile:
            self.simulationResults.saveSensorPlotImage(plt, sensorName)
            #plt.savefig(f"{sensorName}_errorplot.png", bbox_inches ='tight')
            #plt.close(fig) 
        else:
            plt.show()
    
    def plotSelectedVsTrue(self, selectedActivityName, tiPlot, tfPlot, figsize = (400,10), top = 2, toFile = False, windowLength = 1):
        predfreq = self.groundTruthFreq / windowLength

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        # find the slices of the groudtruth
        idx = pd.IndexSlice
        trueLablesSequence = self.imuSensorsDataFrame.loc[idx[self.person, self.session, :], idx['labels', self.activityCategory]].values[tiPlot:tfPlot]
        trueMultiLabelSequence = MultiLabelSequence(trueLablesSequence)
        trueSlices, trueLabels = trueMultiLabelSequence.getSlicesAndLabelsLists()

        # find the slices of the predicted labels
        activityNumId = [self.activities.dict[self.activityCategory][activityName] for activityName in selectedActivityName]
        predictedMultiLabelSequence = MultiLabelSequence(activityNumId[(tiPlot-self.tiSim)//windowLength:(tfPlot-self.tiSim)//windowLength])
        predictedSlices, predictedLabels = predictedMultiLabelSequence.getSlicesAndLabelsLists()

        # plot the groundtruth labels as background colors in the top half of the plot 
        for i, item in enumerate(trueSlices):
            plt.axvspan((item.start + tiPlot)/self.groundTruthFreq, (item.stop + tiPlot)/self.groundTruthFreq, 
                        facecolor=self.colorDict[self.activityCategory][trueLabels[i]], 
                        #alpha=0.3, 
                        ymin=0, ymax=0.5)

        # plot the predicted labels as background colors in the bottom half of the plot 
        for i, item in enumerate(predictedSlices):
            plt.axvspan((item.start/predfreq + tiPlot/self.groundTruthFreq), (item.stop/predfreq + tiPlot/self.groundTruthFreq),                            
                        facecolor=self.colorDict[self.activityCategory][predictedLabels[i]], 
                        #alpha=0.3, 
                        ymin=0.5, ymax=1)

        plt.text(tiPlot/self.groundTruthFreq, 0.75*top, r'PREDICTED', fontsize = 20)
        plt.text(tiPlot/self.groundTruthFreq, 0.25*top, r'TRUE', fontsize = 20)  
                
        major_ticks = np.arange(tiPlot/self.groundTruthFreq, tfPlot/self.groundTruthFreq, 1)
        minor_ticks = np.arange(tiPlot/self.groundTruthFreq, tfPlot/self.groundTruthFreq, 0.5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        plt.grid(b=True, which='both', axis='x')

        plt.xlim(left=tiPlot/self.groundTruthFreq)
        plt.ylim(top=top, bottom = 0)
        
        plt.xlabel('seconds')

        markers = []
        markerLabels = []
        for activityName, activityNumId in self.activities.dict[self.activityCategory].items():
            color = self.colorDict[self.activityCategory][activityNumId]
            markers.append(plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle=''))
            markerLabels.append(activityName)
        
        plt.legend(markers, markerLabels, numpoints=1, loc = 'upper left')


        if toFile:
            self.simulationResults.saveMaxVotingPlotImage(plt)
            # plt.savefig(f"{self.activityCategory}_SelectedVsTrue.png", bbox_inches ='tight')
        else:
            plt.show()

class SimulationResults(object):
    def __init__(self, activityCategory, person, session, windowLength, lookback, baseDir = ''):
                
        self.rootPath = os.path.join(baseDir, 'simulation_results', activityCategory, f'{person}{session}_window{windowLength}_lb{lookback}') 
        try:
            os.makedirs(self.rootPath)
        except FileExistsError:
            # directory already exists
            pass
    
    def saveSensorConfusionMatrixDf(self, confusionMatrix, sensorName):
        # Save to sensor filepath
        folderPath = os.path.join(self.rootPath, "confusion_matrix")
        try:
            os.makedirs(folderPath)
        except FileExistsError:
            # directory already exists
            pass
        
        filepath =  os.path.join(folderPath, f"{sensorName}_confusion_matrix.csv")
        confusionMatrix.to_csv(filepath)
    
    def saveMaxVotingConfusionMatrixDf(self, confusionMatrix):
        # Save to sensor filepath
        folderPath = os.path.join(self.rootPath, "confusion_matrix")
        try:
            os.makedirs(folderPath)
        except FileExistsError:
            # directory already exists
            pass
        
        filepath =  os.path.join(folderPath, 'max_voting_confusion_matrix.csv')
        confusionMatrix.to_csv(filepath)
    
    def saveSensorPlotImage(self, plt, sensor):
        # Save to sensor filepath
        folderPath = os.path.join(self.rootPath, "plot_images")
        try:
            os.makedirs(folderPath)
        except FileExistsError:
            # directory already exists
            pass
        
        filepath =  os.path.join(folderPath, f'{sensor}_plot_predicted_vs_true.png')
        plt.savefig(filepath, bbox_inches ='tight')
        
    
    def saveMaxVotingPlotImage(self, plt):
        # Save to sensor filepath
        folderPath = os.path.join(self.rootPath, "plot_images")
        try:
            os.makedirs(folderPath)
        except FileExistsError:
            # directory already exists
            pass
        
        filepath =  os.path.join(folderPath, 'max_voting_plot_predicted_vs_true.png')
        plt.savefig(filepath, bbox_inches ='tight')
       

    

    
