import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

class BayesianNN:
        '''
        A class that defines a Bayesian Neural Network (BNN) model to recover the parameters of a cosine series acoustic surface.

        Works on sum of cosine waves. Units are metres for amplitude and wavelength, and radians/pi for phase.
        '''

        def __init__(self, nHidden):
            '''
            Inputs:
            nHidden: The number of hidden layers
            '''
            self.nHidden = nHidden
            self.currentX = None
            self.currentY = None
            self.name = "bnn"

        def _buildModel(self, penaltyMask=1.0):
            '''
            Builds the BNN model.
            
            Parameters:
                penaltyMask (pytensor.tensor): Boolean array that allows some values to be negative.

            Returns:
                pymc.Model: The model object.
            '''

            with pm.Model() as self.model:

                W1 = pm.Normal("W1", mu=0, sigma=1, shape=(self.nInputs, self.nHidden))
                b1 = pm.Normal("b1", mu=0, sigma=1, shape=(self.nHidden,))
                
                W2 = pm.Normal("W2", mu=0, sigma=1, shape=(self.nHidden, self.nHidden))
                b2 = pm.Normal("b2", mu=0, sigma=1, shape=(self.nHidden,))

                W3 = pm.Normal("W3", mu=0, sigma=1, shape=(self.nHidden, self.nHidden))
                b3 = pm.Normal("b3", mu=0, sigma=1, shape=(self.nHidden,))

                W4 = pm.Normal("W5", mu=0, sigma=1, shape=(self.nHidden, self.n_outputs))
                b4 = pm.Normal("b5", mu=0, sigma=1, shape=(self.n_outputs,))
                
                # Input scatter data
                X = pm.Data("X", self.currentX.to_numpy())
                
                # Forward pass
                hidden1 = pm.math.tanh(pt.dot(X, W1) + b1)
                hidden2 = pm.math.tanh(pt.dot(hidden1, W2) + b2)
                hidden3 = pm.math.tanh(pt.dot(hidden2, W3) + b3)
                output = pt.dot(hidden3, W4) + b4

                # Penalize negative values for mu with exponential dropoff based on how many negative values exist
                # Allow some to be negative if pass the penalty mask (e.g. (1,1,0) would let the third value be negative)
                alpha = 10  # Controls steepness; lower values make penalty increase more slowly
                neg_values = pt.minimum(output, 0) * penaltyMask.T
                neg_magnitude = -pt.sum(neg_values, axis=1)
                penalty = pt.exp(-alpha * neg_magnitude)
                pm.Potential("negative-mu-penalty", pm.math.log(penalty))

                # Split observed data
                y_linear = pm.Data("y_linear", self.currentY.to_numpy())

                # Likelihoods
                sigma = pm.HalfNormal("sigma", sigma=0.1, shape=(self.n_outputs,))
                pm.TruncatedNormal("param", lower=(1.0 - penaltyMask)*-10.0, mu=output, sigma=sigma, observed=y_linear)
        
        def train(self, trainX, trainY, burnInCount=2000, sampleCount=5000, penaltyMask=None):
            '''
            Trains the BNN model on the X and Y data.

            Parameters:
                trainX (np.array): The predictor training data (usually scatter response)
                trainY (np.array): The dependent training data (usually surface parameters)
                burnInCount (int): The number of burn in iterations. Will be discarded.
                sampleCount (int): The number of samples to run the chain for.
                penaltyMask (pytensor.tensor): Boolean array that allows some values to be negative.
            '''
            self.currentX = trainX
            self.currentY = trainY
            self.nInputs = trainX.shape[1]
            self.n_outputs = trainY.shape[1]
            if penaltyMask == None:
                penaltyMask = pt.as_tensor_variable(1.0 + np.zeros(shape=(self.currentY.shape[1],)))
            print("Penalty mask: ", penaltyMask)

            self._buildModel(penaltyMask=penaltyMask)
            with self.model:

                az.rcParams['plot.max_subplots'] = 40

                self.trace = pm.sample(draws=sampleCount, tune=burnInCount, chains=1, nuts_sampler="numpyro", return_inferencedata=True)
                self.trace.to_netcdf("results/" + self.name + ".nc")

                summary = az.summary(self.trace)
                print(summary)

        def loadTrace(self):
            '''
            Loads the trace data into the class.
            '''
            self.trace = az.from_netcdf("results/" + self.name + ".nc")

        def getTrace(self):
             '''
             Gets the trace data from the class.

             Return:
                Arviz-Trace: The posterior distributions over the weights
             '''
             return self.trace
        
        def predict(self, testX, testY, penaltyMask=None):
            '''
            Generates a predictive posterior for the test values.
            Currently needs a y array with the same parameter count as training and same length as the test x, however this is not used.

            Parameters:
                testX (np.array): The predictor test data (usually scatter response)
                testY (np.array): The dependent test data (usually surface parameters)
                penaltyMask (pytensor.tensor): Boolean array that allows some values to be negative.

            Returns:
                Arviz-Trace: The posterior predictive distribution
            '''
            
            # TODO: allow without a new Y, doesn't impact the results but is needed for dimensions
            self.currentX = testX
            self.currentY = testY
            self.nInputs = testX.shape[1]
            self.n_outputs = testY.shape[1]
            if penaltyMask == None:
                penaltyMask = pt.as_tensor_variable(1.0 + np.zeros(shape=(self.currentY.shape[1],)))
            print("Penalty mask: ", penaltyMask)
            self._buildModel(penaltyMask=penaltyMask)
            with self.model:
                self.trace = az.from_netcdf("results/" + self.name + ".nc")
                posteriorPredictive = pm.sample_posterior_predictive(self.trace)["posterior_predictive"]
            return posteriorPredictive
        
        def setName(self, name):
             self.name = name