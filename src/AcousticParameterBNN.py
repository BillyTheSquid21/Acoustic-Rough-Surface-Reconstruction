import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt

class BayesianNN:
        def __init__(self, n_hidden):
            self.n_hidden = n_hidden
            self.current_x = None
            self.current_y = None
            self.name = "bnn"
            plt.style.use('science')
            plt.rcParams["font.family"] = "Bitstream Charter"

        def build_model(self):
            with pm.Model() as self.model:
                # Prior distributions for weights and biases
                W1 = pm.Normal("W1", mu=0, sigma=1, shape=(self.n_inputs, self.n_hidden))
                b1 = pm.Normal("b1", mu=0, sigma=1, shape=(self.n_hidden,))
                
                W2 = pm.Normal("W2", mu=0, sigma=1, shape=(self.n_hidden, self.n_outputs))
                b2 = pm.Normal("b2", mu=0, sigma=1, shape=(self.n_outputs,))
                
                #W3 = pm.Normal("W3", mu=0, sigma=1, shape=(self.n_hidden, self.n_hidden))
                #b3 = pm.Normal("b3", mu=0, sigma=1, shape=(self.n_hidden,))
                
                #W3 = pm.Normal("W3", mu=0, sigma=1, shape=(self.n_hidden, self.n_outputs))
                #b3 = pm.Normal("b3", mu=0, sigma=1, shape=(self.n_outputs,))
                
                # Input scatter data
                X = pm.Data("X", self.current_x.to_numpy())
                
                # Forward pass
                hidden1 = pm.math.tanh(pt.dot(X, W1) + b1)
                #hidden2 = pm.math.tanh(pt.dot(hidden1, W2) + b2)
                #hidden3 = pm.math.tanh(pt.dot(hidden2, W3) + b3)
                output = pt.dot(hidden1, W2) + b2
                
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

                az.rcParams['plot.max_subplots'] = 40

                self.trace = pm.sample(draws=sampleCount, tune=burnInCount, chains=1, nuts_sampler="numpyro", return_inferencedata=True)
                self.trace.to_netcdf("results/" + self.name + ".nc")

                az.plot_trace(self.trace, var_names=["W1"], divergences=False, compact=False, combined=False)
                fig = plt.gcf()
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.5, wspace=0.3)
                plt.savefig("results/" + self.name + "_w1_trace.png")

                az.plot_trace(self.trace, var_names=["b1"], divergences=False, compact=False, combined=False)
                fig = plt.gcf()
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.5, wspace=0.3)
                plt.savefig("results/" + self.name + "_b1_trace.png")

                az.plot_trace(self.trace, var_names=["W2"], divergences=False, compact=False, combined=False)
                fig = plt.gcf()
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.5, wspace=0.3)
                plt.savefig("results/" + self.name + "_w2_trace.png")

                az.plot_trace(self.trace, var_names=["b2"], divergences=False, compact=False, combined=False)
                fig = plt.gcf()
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.5, wspace=0.3)
                plt.savefig("results/" + self.name + "_b2_trace.png")

                #az.plot_trace(self.trace, var_names=["W3"], divergences=False, compact=False, combined=False)
                #fig = plt.gcf()
                #fig.tight_layout()
                #fig.subplots_adjust(hspace=0.5, wspace=0.3)
                #plt.savefig("results/" + self.name + "_w3_trace.png")

                #az.plot_trace(self.trace, var_names=["b3"], divergences=False, compact=False, combined=False)
                #fig = plt.gcf()
                #fig.tight_layout()
                #fig.subplots_adjust(hspace=0.5, wspace=0.3)
                #plt.savefig("results/" + self.name + "_b3_trace.png")

                #az.plot_trace(self.trace, var_names=["W4"], divergences=False, compact=False, combined=False)
                #fig = plt.gcf()
                #fig.tight_layout()
                #fig.subplots_adjust(hspace=0.5, wspace=0.3)
                #plt.savefig("results/" + self.name + "_w4_trace.png")

                #az.plot_trace(self.trace, var_names=["b4"], divergences=False, compact=False, combined=False)
                #fig = plt.gcf()
                #fig.tight_layout()
                #fig.subplots_adjust(hspace=0.5, wspace=0.3)
                #plt.savefig("results/" + self.name + "_b4_trace.png")

                summary = az.summary(self.trace)
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