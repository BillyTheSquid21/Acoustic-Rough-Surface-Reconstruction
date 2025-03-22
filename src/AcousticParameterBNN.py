import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

class BayesianNN:
        def __init__(self, n_hidden):
            self.n_hidden = n_hidden
            self.current_x = None
            self.current_y = None
            self.name = "bnn"

        def build_model(self):
            with pm.Model() as self.model:
                # Prior distributions for weights and biases
                W1 = pm.Normal("W1", mu=0, sigma=1, shape=(self.n_inputs, self.n_hidden))
                b1 = pm.Normal("b1", mu=0, sigma=1, shape=(self.n_hidden,))
                
                W2 = pm.Normal("W2", mu=0, sigma=1, shape=(self.n_hidden, self.n_hidden))
                b2 = pm.Normal("b2", mu=0, sigma=1, shape=(self.n_hidden,))
                
                W3 = pm.Normal("W3", mu=0, sigma=1, shape=(self.n_hidden, self.n_hidden))
                b3 = pm.Normal("b3", mu=0, sigma=1, shape=(self.n_hidden,))
                
                W4 = pm.Normal("W4", mu=0, sigma=1, shape=(self.n_hidden, self.n_outputs))
                b4 = pm.Normal("b4", mu=0, sigma=1, shape=(self.n_outputs,))
                
                # Input scatter data
                X = pm.Data("X", self.current_x.to_numpy())
                
                # Forward pass
                hidden1 = pm.math.tanh(pt.dot(X, W1) + b1)
                hidden2 = pm.math.tanh(pt.dot(hidden1, W2) + b2)
                hidden3 = pm.math.tanh(pt.dot(hidden2, W3) + b3)
                output = pt.dot(hidden3, W4) + b4
                
                # Input param data
                y = pm.Data("y", self.current_y.to_numpy())

                sigma = pm.HalfNormal("sigma", sigma=1, shape=(self.n_outputs,))
                pm.Normal("param", mu=output, sigma=sigma, observed=y)
        
        def train(self, train_x, train_y, burnInCount=2000, sampleCount=5000):
            self.current_x = train_x
            self.current_y = train_y
            self.n_inputs = train_x.shape[1]
            self.n_outputs = train_y.shape[1]
            self.build_model()
            with self.model:
                self.trace = pm.sample(draws=sampleCount, tune=burnInCount, chains=1, nuts_sampler="numpyro", return_inferencedata=True)
                self.trace.to_netcdf("results/" + self.name + ".nc")

                summary = az.summary(self.trace, circ_var_names=["phase"])
                print(summary)

        def loadTrace(self):
             self.trace = az.from_netcdf("results/" + self.name + ".nc")

        def getTrace(self):
             return self.trace
        
        def predict(self, test_x, test_y):
            # TODO: allow without a new Y, doesn't impact the results but is needed for dimensions
            self.current_x = test_x
            self.current_y = test_y
            self.n_inputs = test_x.shape[1]
            self.n_outputs = test_y.shape[1]
            self.build_model()
            with self.model:
                pm.set_data({"X": test_x})
                pm.set_data({"y": test_y})
                self.trace = az.from_netcdf("results/" + self.name + ".nc")
                posterior_predictive = pm.sample_posterior_predictive(self.trace)["posterior_predictive"]
            return posterior_predictive
        
        def setName(self, name):
             self.name = name