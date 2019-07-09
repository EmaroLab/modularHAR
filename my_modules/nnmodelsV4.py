import numpy
import keras
import os
import pickle

from keras.models import Model
from keras import layers
from keras import Input

from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback

from abc import ABC, abstractmethod

class NNModel(object):
    """ 
    A class used to wrap a keras model
    
    Attributes 
    ----------
    model : keras model
        the keras model (regressor) this class wraps
    callbackList : list
        the list of callbacks used when fitting the model
    lookback : int
        the number of timeseries samples used by regerssor model atrribute to make a prediction
    history : keras object
        the training history object from keras modules 
    trainMean : numpy array
        each element of the array is the mean of of all timesteps of a feature
        of the all the multivariate timeseries that constitute the training data relative to the
        single activity for which the model attribute is a regressor
    trainStd : numpy array
        each element of the array is the standard deviation of of all timesteps of a feature
        of the all the multivariate timeseries that constitute the training data relative to the
        single activity for which the model attribute is a regressor
    identifier : dictionary
        used to identify the model with keys activityCategory, sensor, activityName
    sensorChannels : int
        number of sensors per IMU
    errorMaxThreshold : float
        if the prediction error is higher than this threshold it means that the activity 
        being performed is not the activity the instance of NNmodel represents  


    Methods
    -------
    setLookback(lookback)
    
    setTrainMeanAndStd(trainMean, trainStd)

    compile(modelOptimizer = 'adam', modelLoss = 'mae')
        Compliles the keras model
    fit(trainSamples = None, trainTargets = None, fitParams = None)
        Trains the keras model
    loadFromFilepath(modelFilepath)
        Sets the model attribute to the keras model at specified filepath 
    evaluate(testSamples, testTargets)
        Returns the prediction error of the keras model attribute 
    predict(testSamples)
        Returns the predicted values, uses the predict method of the keras model class
    """


    def __init__(self, model, callbacksList, identifier = None, lookback = None):
        """"
        Parameters
        ----------
        model : keras model
        callbackList : list
            the list of callbacks
        """

        self.model = model
        self.callbacksList = callbacksList
        self.identifier = {
               'activityCategory' : identifier['activityCategory'],
               'sensor' : identifier['sensor'],
               'activityName' : identifier['activityName'],
            }
        self.sensorChannels = model.layers[0].input_shape[-1]
        self.lookback = lookback
        self.trainMean = None
        self.trainStd = None
        self. errorMaxThreshold = None

        self.model.summary()

    def setLookback(self, lookback):
        self.lookback = lookback

    # train mean and std are necessary when evaluating on samples that are not standardized
    def setTrainMeanAndStd(self, trainMean, trainStd):
        """  
        Parameters
        ----------
        trainMean : numpy array
            each element of the array is the mean of of all timesteps of a feature
            of the all the multivariate timeseries that constitute the training data relative to the
            single activity for which the model attribute is a regressor
        trainStd : numpy array
            each element of the array is the standard deviation of of all timesteps of a feature
            of the all the multivariate timeseries that constitute the training data relative to the
            single activity for which the model attribute is a regressor
        """

        self.trainMean = trainMean
        self.trainStd = trainStd

    def compile(self, modelOptimizer = 'adam', modelLoss = 'mae'):
        self.model.compile(optimizer=modelOptimizer, loss=modelLoss)

    def fit(self, trainSamples = None, trainTargets = None, fitParams = None):
        """Trains the keras model 
        
        Parameters
        ----------
        trainSamples : numpy array 
            the data used to train the keras model shape = (samples, lookback, features)
        trainTargets : numpyarray
            the targets corresponding to the traing data shape = (samples, features)
        fitPrams: dictionary
            contains the information necessary to train the model, its key values are eqivalent
            to the fit method parameter or the keras module  
            """
        self.history = self.model.fit( trainSamples, trainTargets,
                                       shuffle=fitParams['shuffle'],
                                       epochs=fitParams['epochs'],
                                       batch_size=fitParams['batch_size'],
                                       callbacks=self.callbacksList,
                                       validation_split=fitParams['validation_split'])

    def loadFromFilepath(self, modelFilepath):
        """Sets the model attribute to the keras model at specified filepath
        
        Parameters
        ----------
        modelFilepath : str
            colmpete filepath to the keras model
        """
        
        self.model = keras.models.load_model(modelFilepath)

    def evaluate(self, testSamples, testTargets):
        """ Returns the prediction error

        When multiple samples and targets are evaluated it returns the mean off all errors.
        If only one sample is given the shape of the parameters must be the same as
        when multiple samples are evaluated

        Parameters
        ----------
        testSamples : numpy array 
            shape = (samples, lookback, features)
        testTargets : numpyarray
            shape = (samples, features)
        
        Returns
        -------
        int
            the mean of the prediction errors
        """

        return self.model.evaluate(testSamples, testTargets, verbose = 0)
    
    def predict(self, testSamples):
        """ Returns the predicted np.array, uses the predict method of the keras model class

        Parameters
        ----------
        testSamples : numpy array 
            shape = (samples, lookback, features)
                
        Returns
        -------
        numpy array
            shape = (samples, features)
        """    
        
        return self.model.predict(testSamples)

# Model Factory Interface
class ModelFactory(ABC):
    def getModel(self):
        pass

# Contrete Model Factory
class LSTMModelFactory(ModelFactory):
    """ A factory used to create LSTM keras models """
    def __init__(self, params):
        """
        Parameters
        ----------
        params : dict
            a dictionary used to specify the model parameters, the dictionary keys
            are equivalent to the keras Input and LSTM parameters
        """
        
        self.params = params
    
    def getModel(self):
        input_tensor = Input(shape=self.params['input_shape'])
        return_sequences = self.params['layers'] > 1
        x = layers.LSTM(self.params['neurons'],
                        activation=self.params['activation'],
                        recurrent_activation=self.params['recurrent_activation'],
                        dropout=self.params['dropout'], 
                        recurrent_dropout=self.params['recurrent_dropout'],
                        return_sequences = return_sequences)(input_tensor)
        for i in range(self.params['layers'] - 1):
            return_sequences = i < self.params['layers'] - 2
            x = layers.LSTM(self.params['neurons'],
                        activation=self.params['activation'],
                        recurrent_activation=self.params['recurrent_activation'], 
                        dropout=self.params['dropout'], 
                        recurrent_dropout=self.params['recurrent_dropout'],
                        return_sequences = return_sequences)(x)
        output_tensor = layers.Dense(self.params['input_shape'][-1])(x)

        return Model(input_tensor, output_tensor)
   
class CallbacksListFactory(ABC):
    def getList(self, modelFilepath):
        pass

class BaseCallbacksListFactory(CallbacksListFactory):
    def getList(self, modelFilepath):
        callbacksList = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=6,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=modelFilepath,
                monitor='val_loss',
                save_best_only=True,
            ),
        ] 

        return callbacksList

class ColabCallbacksListFactory(CallbacksListFactory):
    def getList(self, modelFilepath):
        tbc=TensorBoardColab()
        self.callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=modelFilepath,
                monitor='val_loss',
                save_best_only=True,
            ),
            TensorBoardColabCallback(tbc),
        ]

def saveNNModel(nnModel, baseDir = None):

    activityCategory = nnModel.identifier['activityCategory']
    sensor = nnModel.identifier['sensor']
    activityName = nnModel.identifier['activityName']

    directoryPath = os.path.join(baseDir, activityCategory, sensor)
    fileName = f"{sensor}_{activityName}_lb{nnModel.lookback}_senCh{nnModel.sensorChannels}.NNModel"
    filepath = os.path.join(directoryPath, fileName)
    
    try:
        os.makedirs(directoryPath)
    except FileExistsError:
        # directory already exists
        pass
    
    with open(filepath, 'wb') as filehandler:
        pickle.dump(nnModel, filehandler, pickle.HIGHEST_PROTOCOL)

    print(f"\nNNModel object saved to {filepath}")

def loadNNModel(identifier, lookback = 30, sensorChannels = 6, baseDir = None):
    directoryPath = os.path.join(baseDir, identifier['activityCategory'], identifier['sensor'])
    fileName = f"{identifier['sensor']}_{identifier['activityName']}_lb{lookback}_senCh{sensorChannels}.NNModel"
    filepath = os.path.join(directoryPath, fileName)

    with open(filepath, 'rb') as filehandler:
        return pickle.load(filehandler)


    