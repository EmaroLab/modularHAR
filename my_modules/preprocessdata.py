import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from orderedset import OrderedSet

class IMUSensors(object):
    """
    A class used to represent all IMUSensors
    
    Attributes
    ----------
    df : pandas dataframe
        a dataframe wich contains all the IMU timesteps values. 
        The first level of the column multiindex indicates the IMU sensor and the labels,
        The second level of the column multiindex indicates the specific 
        sensor of the corresponding first level IMU and the specific label cataegory.
        The first level of the row multindex indicates the Person,
        the second level of the row multiindex indicates the session,
        each row is a time step  
    
    sensorNames : list 
        the list of the IMU sensor names

    Methods
    -------
    singleSensorDf(sensorName)
        Returns the dataframe corresponding to a single IMU sensor specified by sensorName

    """
    
    def __init__(self, allSensorsDf):
        """
        Parameters
        ----------
        allSensorsDf : pandas dataframe
            a dataframe wich contains all the IMU timesteps values. 
            The first level of the column multiindex indicates the IMU sensor,
            The second level of the column multiindex indicates the specific 
            sensor of the corresponding first level IMU.
            The first level of the row multindex indicates the Person,
            the second level of the row multiindex indicates the session,
            each row is a time step  
        """
        self.df = allSensorsDf
        self.sensorNames = [sensor for sensor in OrderedSet(allSensorsDf.columns.get_level_values(0)) if sensor != 'labels']


    def singleSensorDf(self, sensorName):
        """Returns the dataframe corresponding to a single IMU sensor specified by sensorName
        
        Parameters
        ----------
        senosrName : str
            the name of the IMU sensor as specified in the first level of the df attribute column multiindex
        
        Returns
        -------
            the dataframe corresponding to a single IMU sensor specified by sensorName with corresponding labels
        """

        return self.df[[sensorName,'labels']]

class Activities(object):
    """ 
    a class used to represent the label encoding of the opportunity dataset
    
    Attributes
    ----------
    dict : dictionary
        the multilevel dictionary whose first level keys are the names of the 
        activity categories, the second level keys are the activity names and the
        corresponding values are the numerical encoding used in the opportunity dataset

    Methods
    -------
    getNamesWithCategory(activityCategory)
        Returns the list of all activity names corresponding to activityCatgeory
        (first level key of dict attribute), except for null activity

    """

    def __init__(self):
        self.dict = {
            'BothArmsLabel' : {
                'nullActivity' : 0,
                'Walk' : 1,
                'SitDown' : 2,
                'StandUp' : 3,
                'OpenDoor' : 4,
                'CloseDoor' : 5,
                'PourWater' : 6,
                'DrinkGlass' : 7,
                'BrushTeeth' : 8,
                'CleanTable' : 9,
                    },

            'RightArmLabel' : {
                'nullActivity' : 0,
                'Walk' : 1,
                'SitDown' : 2,
                'StandUp' : 3,
                'OpenDoor' : 4,
                'CloseDoor' : 5,
                'PourWater' : 6,
                'DrinkGlass' : 7,
                'BrushTeeth' : 8,
                'CleanTable' : 9,
                    },

            'LeftArmLabel' : {
                'nullActivity' : 0,
                'Walk' : 1,
                'SitDown' : 2,
                'StandUp' : 3,
                #'OpenDoor' : 4,
                #'CloseDoor' : 5,
                'PourWater' : 6,
                'DrinkGlass' : 7,
                #'BrushTeeth' : 8,
                'CleanTable' : 9,
                    },
            
            'Locomotion' : {
                'nullActivity' : 0,
                'Walk' : 1,
                'SitDown' : 2,
                'StandUp' : 3,
                },
        }


    def getNamesWithCategory(self, activityCategory):
        """Returns the list of all activity names corresponding to activityCatgeory, except for null activity
        
        Parameters
        ----------
        activityCategory : str
            first level key of the dict attribute"""
        
        return [key for key in self.dict[activityCategory].keys() if key != 'nullActivity']

class MultiLabelSequence(object):
    """ a class used to represent a sequence of sequences of numbers, 
    
    a sequence of sequences of numbers is like a multilabel sequence 
    where a label is repeated multiple times then another label is repeated multiple times, 
    then another and so on. The same label can be present in different subsequences
    
    Attributes
    ----------
    sequence : list or numpy array vector
        the sequence of sequences of numbers for example [0,0,0,1,1,0,0,0,0,2,2,2,5,5,5]
    indecesWhereSequenceChanges : list of int
        indeces where the number indicating a specific subsequence changes, in the previous
        example [2,4,8,11,13]
    rangesWhereSequencesChanges : list of int
        numbers used to indicate the ranges where the number indicating a specific 
        subsequence changes, in the previous example [0,3,5,9,12,14]  
    slices: list of slices
        the list of slices each one of them representing a different subsequence of the same numbers
    labels : list of int
        the list of labels corresponding to the list of slices in the slice attribute

    Methods
    -------
    findIndecesWhereSequenceChanges()
        the method used to find the attribute indecesWhereSequenceChanges
    findRangesWhereSequencesChanges()
        the method used to find the attribute rangesWhereSequencesChanges
    getLabelsAndSlicesLists()
        returns the list of slices and the corresponding list of labels by splitting the sequence 
        of sequences in the sequence atttribute
    getSlicesWithLabel(label)
        returns only the list of slices corresponding to label specified as a pareameter by 
        splitting the sequence of sequences in the sequence atttribute
    """

    def __init__(self,sequence):
        self.sequence = sequence
        self.indecesWhereSequenceChanges = None
        self.rangesWhereSequencesChanges = None
        self.slices = None
        self.labels = None
      
        self.findIndecesWhereSequenceChanges()
        self.findRangesWhereSequencesChanges()

    def findIndecesWhereSequenceChanges(self):
        """the method used to find the attribute indecesWhereSequenceChanges
        
        for example give the sequence of sequences [0,0,0,1,1,0,0,0,0,2,2,2,5,5,5] this method
        finds the indeces where the number indicating a specific subsequence changes, in the 
        example these indeces are [2,4,8,11,13]
        """
        sequenceChange = np.array(self.sequence[:-1]) != np.array(self.sequence[1:])
        sequenceChange = np.append(sequenceChange, True)
        self.indecesWhereSequenceChanges = np.array(range(len(sequenceChange)))[sequenceChange]

    def findRangesWhereSequencesChanges(self):
        """ the method used to find the attribute rangesWhereSequencesChanges 
        
        for example given the indecesWhereSequenceChanges [2,4,8,11,13] this method finds the 
        numbers that can be used to indicate the ranges where subsequences change, in this 
        example: [0,3,5,9,12,14]
        """
        self.rangesWhereSequencesChanges = [0] + list(self.indecesWhereSequenceChanges + 1) # first range starts at zero, ranges ends the index before to the stop index

    def getSlicesAndLabelsLists(self):
        """returns the list of slices and the corresponding list of labels 
        
        this is done by splitting the sequence attribute into the different subsequences 

         Returns
         -------
         list of slice objects
         list of int
         """
        numberOfDifferentSequences = len(self.rangesWhereSequencesChanges) - 1

        self.slices = [slice(0)] * numberOfDifferentSequences
        self.labels = [int()] * numberOfDifferentSequences
        for i in range(numberOfDifferentSequences):
            self.slices[i] = slice(self.rangesWhereSequencesChanges[i], self.rangesWhereSequencesChanges[i+1])
            self.labels[i] = self.sequence[self.rangesWhereSequencesChanges[i]]

        return self.slices,  self.labels

    def getSlicesWithLabel(self, label):
        """ returns only the list of slices corresponding to label specified as a parameter 
        
        this is done by splitting the sequence attribute into the different subsequences and looking for 
        a particular label
        
        Parameters
        ----------
        label : int
            the number (label) that characterize the subsequences 
        
        Returns
        -------
        list of slice objects
        """
        numberOfDifferentSequences = len(self.rangesWhereSequencesChanges) - 1

        slices = []
        for i in range(numberOfDifferentSequences):
            if self.sequence[self.rangesWhereSequencesChanges[i]] == label:
                slices.append(slice(self.rangesWhereSequencesChanges[i], self.rangesWhereSequencesChanges[i+1]))

        return slices

class Timeseries(object):
    """ a class that represent multiple timeseries corresponding to a single IMU sensor and a single activity 
    
    Attributes
    ----------
    items : list of pandas datframe
        the dataframes represents timeseries corresponding to a single activity,
        the rows are the timesteps
    sensorChannels : int
        number of sensors per IMU
    concatDf : pandas dataframe
        the dataframe obtained by concatenating all the datframes in itmes along the row axis that
        represents the timesteps  
    normItems : list of pandas dataframe
        the same datframes in items but with the columns normalized with respect to the corresponding 
        columns in the concatDf attribute.
    standItems : list of pandas dataframe
        the same datframes in items but with the columns standardized with respect to the corresponding 
        columns in the concatDf attribute.

    Methods
    -------
    getConcatenatedDf()
        returns the dataframe obtained by concatenating all dataframes in the items attribute, along the row axis
    normalizeTsItems(maxValue = None, minValue = None)
        normalizes the columns of the dataframes in the items attribute 
    standardizeTsItems(meanValue = None, stdValue = None)
        standardizes the columns of the dataframes in the items attribute 
    plotWithIndexAndChannels(tsIndex, channelColumns = None, scaling = None)
        plots the columns of a timeseries dataframe  
    """

    def __init__(self, dataframe, tsSlices, sensorChannels = 6):
        """
        Parameters 
        ----------
        dataframe : pandas dataframe
            a dataframe corresponding to a single IMU sensor with labels columns
        tsSlices : list of slice objects
            the slices in the list indicate the timesteps indeces' range corresponing to a specific activity
        sensorChannels : int, optional
            number of sensors per IMU 
        """
        #get list of dataframes form a dataframe and a list of row slices
        self.items = []
        for tsSlice in tsSlices:
            self.items.append(dataframe.iloc[tsSlice,:])

        self.sensorChannels = sensorChannels
        self.concatDf = None
        self.normItems = []
        self.standItems = []


    def getConcatenatedDf(self):
        """returns the dataframe obtained by concatenating all dataframes in items attribute along the row axis"""
        if self.concatDf is None:
            self.concatDf = pd.concat(self.items)
        return self.concatDf


    def normalizeTsItems(self, maxValue = None, minValue = None):
        self.normItems = []
        """normalizes the columns of the dataframes in the items attribute 
        
        if no maxValue and no minValue are specified the normalization is performed 
        with respect to the corresponding columns in concatDf attribute
        
        Parameters
        ---------- 
        maxValue : np.array (row vector)
            the shape depends on the columns correspondig to the IMU sensor (self.sensorChannels,) 
        
        minValue : np.array (row vector)
        """

        if maxValue is not None and minValue is not None:
            maxVal = maxValue
            minVal = minValue

        else:
            if self.concatDf is None:
                self.getConcatenatedDf()
            maxVal = self.concatDf.iloc[:,:self.sensorChannels].max()
            minVal = self.concatDf.iloc[:,:self.sensorChannels].min()

        if not self.normItems:
            for df in self.items:
                normDf = df.iloc[:,:self.sensorChannels]   # select only the sensor columns
                normDf = (normDf-minVal)/(maxVal-minVal)
                normDf = pd.concat([normDf, df.iloc[:,-4:]], axis = 1)   # concatenate the label columns
                self.normItems.append(normDf)


    def standardizeTsItems(self, meanValue = None, stdValue = None):
        """standardizes the columns of the dataframes in the items attribute 
        
        if no meanValue and no stdValue are specified the normalization is performed 
        with respect to the corresponding columns in concatDf attribute

        Parameters
        ---------- 
        meanValue : np.array (row vector)
            the shape depends on the columns correspondig to the IMU sensor (self.sensorChannels,) 
        
        stdValue : np.array (row vector)
        """

        self.standItems = []
        if meanValue is not None and stdValue is not None:
            meanVal = meanValue
            stdVal = stdValue
        else:
            if self.concatDf is None:
                self.getConcatenatedDf()
            meanVal = self.concatDf.iloc[:,:self.sensorChannels].mean()
            stdVal = self.concatDf.iloc[:,:self.sensorChannels].std()

        if not self.standItems:
            for df in self.items:
                standDf = df.iloc[:,:self.sensorChannels]   # select only the sensor columns
                standDf=(standDf-meanVal)/stdVal
                standDf = pd.concat([standDf, df.iloc[:,-4:]], axis = 1)   # concatenate back the label columns
                self.standItems.append(standDf)

            
    def plotWithIndexAndChannels(self, tsIndex, channelColumns = None, scaling = None):
        """plots the columns of a timeseries dataframe
        
        Parameters
        ----------
        tsIndex : int
            the index that specify which dataframe in the items attribute is to be plotted
        channelColumns : list of int, optional
            the columns to be plotted, if no column is specified the first self.sensorChannels columns are plotted
        scaling : string
            'normalized' or 'standardized'
        """
        if scaling == 'normalized':
            sensorValues = self.normItems[tsIndex].values
        elif scaling == 'standardized':
            sensorValues = self.standItems[tsIndex].values
        else:
            sensorValues = self.items[tsIndex].values

        time = range(len(sensorValues))

        if not channelColumns:
            channelColumns = range(self.sensorChannels)

        for i in channelColumns:
            plt.plot(time,  sensorValues[:,i], label='sensor column: {0}'.format(i))

        plt.title('Timeseries')
        plt.xlabel('Sample index')
        plt.ylabel('Sensordata values')
        plt.legend()

        plt.show()
