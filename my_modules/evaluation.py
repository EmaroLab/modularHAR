import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics

class Results(object):
    """a class to obtain classification results of a set of NNModel instances
    
    Attributes
    ----------
    models : list of NNmodel instances (nnmodelsV4 module)
    activityNames : list of str (activity names aka activity labels)
    """

    def __init__(self, models, activityNames):
        self.models = models
        self.activityNames = activityNames
        self.testSamplesList = []  # list of different models' samples np.array
        self.testTargetsList = []  # list of different models' targtes np.array


    def setSamplesAndTargestLists(self, testSamplesList, testTargetsList):
        """sets the list of tests samples and targets

        Parameters
        ----------
        testSamples : list of np.array
        testTargets : list of np.array

        """
        self.testSamplesList = testSamplesList
        self.testTargetsList = testTargetsList


    # predicted errors
    def modelsError(self, testSamples, testTargets):
        """return models' errors in array form

        the prediction error of each model in self.models is evaluated one all
        the samples: the rows corresponds to the samples while the columns to the models

        Parameters
        ----------
        testSamples : np.array
        testTargets : np.array

        Returns
        -------
        np.array
          array of errors one for every NN model in self.models
          shape (number of samples, number of models)
        """

        error = np.zeros((len(testSamples), len(self.models)))
        for i, model in enumerate(self.models):
            predictedValues = model.predict(testSamples)
            error[:,i] =  self.mae(predictedValues, testTargets)
            
        return error


    def mae(self, predictedValues, targetValues):
        """ return np.array of mean absolute values of shape = (samples,)
        
        Parameters
        ----------
        predictedValues : numpy array (shape = (samples, features))
        targetValues : numpy array (shape = (samples, features))

        Returns
        -------
        numpy array (shape = (samples,)) 
        """

        return np.mean(np.abs(targetValues - predictedValues), axis=1)   

    # predicted class (wraper around modelsError)
    def predictedClass(self, testSamples, testTargets):
        """return the predicted class in one hot encoded style

        starting from the error array computed by modelsError,
        for each row find the column with the lowest error
        and assign 1 to that column and zero to the others

        Parameters
        ----------
        testSamples : np.array
        testTargets : np.array
        Returns
        -------
        np.array
          array of predicted class one hot encoded shape (num of samples, num of modles)
          for a given sample (row) the column == 1 is the predicted class other columns are zero
        """

        error = self.modelsError(testSamples, testTargets)
        self.predClass = np.zeros(error.shape, dtype=int)
        self.predClass[range(len(error)), np.argmin(error, axis=1)] = 1  # fancy indexing

        return self.predClass


    def getConfusionMatrix(self):
        """Returns the confusion matrix dataframe 

        the list of self.testSamples and self.testTargetsList
        must be set before running this method
        """

        classArray = np.zeros(len(self.models), dtype=int)
        classArray[0] = 1
        self.actualClass = np.repeat([classArray], len(self.testSamplesList[0]), axis=0)

        self.predClass = self.predictedClass(self.testSamplesList[0], self.testTargetsList[0])

        for i in range(1, len(self.testSamplesList)):
            classArray = np.zeros(len(self.models), dtype=int)
            classArray[i] = 1
            self.actualClass = np.concatenate(
                (self.actualClass, np.repeat([classArray], len(self.testSamplesList[i]), axis=0)))

            self.predClass = np.concatenate(
                (self.predClass, self.predictedClass(self.testSamplesList[i], self.testTargetsList[i])))

        matrix = metrics.confusion_matrix(self.actualClass.argmax(axis=1), self.predClass.argmax(axis=1))
        confusionMatrixDf = pd.DataFrame(matrix.transpose(), index = self.activityNames, columns = self.activityNames)
        confusionMatrixDf.columns.name = 'predicted'
        confusionMatrixDf.index.name = 'true'

        return confusionMatrixDf

    def getEvaluationDf(self):
        evalLoss = np.zeros((len(self.models), len(self.activityNames)))
        for i, model in enumerate(self.models):
            evalLoss[:,i] = np.array([model.evaluate(self.testSamplesList[k], self.testTargetsList[k]) for k in range(len(self.activityNames))]).T

        evalDf = pd.DataFrame(data=evalLoss, index= self.activityNames, columns=self.activityNames)
        evalDf.columns.name = 'models'
        evalDf.index.name = 'data'

        return evalDf

