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

        def build_model(self, penalty_mask=1.0):
            '''
            Penalty mask is used to allow some negative values (e.g. phase)
            '''

            with pm.Model() as self.model:

                W1 = pm.Normal("W1", mu=0, sigma=1, shape=(self.n_inputs, self.n_hidden))
                b1 = pm.Normal("b1", mu=0, sigma=1, shape=(self.n_hidden,))
                
                W2 = pm.Normal("W2", mu=0, sigma=1, shape=(self.n_hidden, self.n_hidden))
                b2 = pm.Normal("b2", mu=0, sigma=1, shape=(self.n_hidden,))

                W3 = pm.Normal("W3", mu=0, sigma=1, shape=(self.n_hidden, self.n_hidden))
                b3 = pm.Normal("b3", mu=0, sigma=1, shape=(self.n_hidden,))

                W4 = pm.Normal("W5", mu=0, sigma=1, shape=(self.n_hidden, self.n_outputs))
                b4 = pm.Normal("b5", mu=0, sigma=1, shape=(self.n_outputs,))
                
                # Input scatter data
                X = pm.Data("X", self.current_x.to_numpy())
                
                # Forward pass
                hidden1 = pm.math.tanh(pt.dot(X, W1) + b1)
                hidden2 = pm.math.tanh(pt.dot(hidden1, W2) + b2)
                hidden3 = pm.math.tanh(pt.dot(hidden2, W3) + b3)
                output = pt.dot(hidden3, W4) + b4

                # Penalize negative values for mu with exponential dropoff based on how many negative values exist
                # Allow some to be negative if pass the penalty mask (e.g. (1,1,0) would let the third value be negative)
                alpha = 10  # Controls steepness; lower values make penalty increase more slowly
                neg_values = pt.minimum(output, 0) * penalty_mask.T
                neg_magnitude = -pt.sum(neg_values, axis=1)
                penalty = pt.exp(-alpha * neg_magnitude)
                pm.Potential("negative-mu-penalty", pm.math.log(penalty))

                # TODO: make work for other configs than 3 param
                # Testing circular
                # Split model outputs
                output_linear = output[:, :2]
                #output_angle = output[:, 2]

                # Split observed data
                y_linear = pm.Data("y_linear", self.current_y.to_numpy()[:, :2])
                #y_angle = pm.Data("y_angle", self.current_y.to_numpy()[:, 2])

                # Likelihoods
                sigma = pm.HalfNormal("sigma", sigma=0.1, shape=(2,))
                pm.TruncatedNormal("param", lower=0.0, mu=output_linear, sigma=sigma, observed=y_linear)

                # Von Mises for angle
                #sigma_offset = pm.HalfNormal("sigma_offset", sigma=0.1, shape=(1,))
                #kappa = 1.0 / (sigma_offset**2)
                #pm.VonMises("offset", mu=output_angle, kappa=kappa, observed=y_angle)
        
        def train(self, train_x, train_y, burnInCount=2000, sampleCount=5000, penalty_mask=None):
            self.current_x = train_x
            self.current_y = train_y
            self.n_inputs = train_x.shape[1]
            self.n_outputs = train_y.shape[1]
            if penalty_mask == None:
                penalty_mask = pt.as_tensor_variable(1.0 + np.zeros(shape=(self.current_y.shape[1],)))
            print("Penalty mask: ", penalty_mask)

            self.build_model(penalty_mask=penalty_mask)
            with self.model:

                az.rcParams['plot.max_subplots'] = 40

                #inference = pm.ADVI()
                #approx = pm.fit(n=50_000, method=inference)
                #self.trace = approx.sample(draws=30_000)

                #plt.figure(figsize=(16,9))
                #plt.plot(approx.hist, label="old ADVI", alpha=0.3)
                #plt.legend()
                #plt.ylabel("ELBO")
                #plt.xlabel("iteration")

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

                az.plot_trace(self.trace, var_names=["W3"], divergences=False, compact=False, combined=False)
                fig = plt.gcf()
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.5, wspace=0.3)
                plt.savefig("results/" + self.name + "_w3_trace.png")

                az.plot_trace(self.trace, var_names=["b3"], divergences=False, compact=False, combined=False)
                fig = plt.gcf()
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.5, wspace=0.3)
                plt.savefig("results/" + self.name + "_b3_trace.png")

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
            penalty_mask = pt.as_tensor_variable(1.0 + np.zeros(shape=(self.current_y.shape[1],)))
            self.build_model(penalty_mask=penalty_mask)
            with self.model:
                self.trace = az.from_netcdf("results/" + self.name + ".nc")
                posterior_predictive = pm.sample_posterior_predictive(self.trace)["posterior_predictive"]
            return posterior_predictive
        
        def setName(self, name):
             self.name = name