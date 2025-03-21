import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import seaborn as sns
import pandas as pd
import pytensor.tensor as pt
from tqdm import tqdm

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    np.random.seed(42)

    # Create the neural network model
    cosine_count = 1

    # Microphone array
    SourceLocation = [-0.20049999999999987,0.21884]
    ReceiverLocationsX =[-0.131500000000000,-0.112000000000000,-0.0924999999999999,
    -0.0719999999999999,-0.0514999999999999,-0.0309999999999999,-0.0104999999999999,0.00750000000000015,
    0.0290000000000001,0.0480000000000001,0.0685000000000001,0.0900000000000001,0.109500000000000,
    0.129000000000000,0.151000000000000,0.166500000000000,0.187500000000000,
    0.208000000000000,0.229500000000000,0.249500000000000,0.268500000000000,
    0.289000000000000,0.310000000000000,0.327500000000000,0.347500000000000,
    0.368500000000000,0.389000000000000,0.409500000000000,0.426500000000000,
    0.448500000000000,0.467000000000000,0.488500000000000,0.508500000000000,
    0.528000000000000]
    ReceiverLocationsY = [0.283024319442090,
                            0.283006369700232,
                            0.283208419958374,
                            0.283436216383600,
                            0.283364012808826,
                            0.283731809234052,
                            0.283969605659278,
                            0.283593036666793,
                            0.283656579259103,
                            0.284565756433703,
                            0.283773552858929,
                            0.283947095451239,
                            0.283819145709381,
                            0.284651195967522,
                            0.284087611643374,
                            0.284616677233179,
                            0.284777346741947,
                            0.285195143167173,
                            0.284838685759483,
                            0.285313609101167,
                            0.285382786275767,
                            0.285680582700993,
                            0.284461252209761,
                            0.284291810133734,
                            0.284866733475418,
                            0.285097402984186,
                            0.286005199409412,
                            0.285352995834638,
                            0.285200680675069,
                            0.286317096350921,
                            0.286443400441979,
                            0.286756943034289,
                            0.287121866375973,
                            0.286673916634115]
    RecLoc = []
    for i in range(len(ReceiverLocationsX)):
        RecLoc.append([ReceiverLocationsX[i],ReceiverLocationsY[i]])

    SourceAngle = np.pi / 3

    frequency = 14_000

    from src.AcousticParameterMCMC import AcousticParameterMCMC
    from src.Directed2DVectorized import Directed2DVectorised
    from src.SymbolicMath import SymCosineSurface

    # TODO: Try only in specular interval
    # TODO: Find spectral point of the central line, take ~10cm on each side 
    # TODO: Get SP for each reciever, if they lie in the interval, that is good

    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, frequency)
    def generate_microphone_pressure(parameters,uSamples=700):
        def newFunction(x):
            return SymCosineSurface(x,parameters).copy()

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor 
    
    def get_spec_point(sourceloc, recloc):
        '''
        Zero index is source, One index is reciever
        '''
        return (recloc[0]*sourceloc[1])/(sourceloc[1]+recloc[1])
    
    # Get point where directivity intersects
    specpoint = SourceLocation[0] + SourceLocation[1]*np.tan((np.pi/2)-SourceAngle)
    specspread = 0.1 # 10 cm each side

    indices = []
    for i in range(len(RecLoc)):
        rspec = get_spec_point(SourceLocation, RecLoc[i])
        if rspec > specpoint - specspread and rspec < specpoint + specspread:
            indices.append(i)

    generate_data = True
    p_count = 250
    train_count = p_count
    
    if generate_data:
        params = []
        scatters = []
        ampspace = np.random.uniform(0.0, 0.01, p_count)
        wlspace = np.linspace(0.05, 0.05, p_count)
        for i in range(p_count):
            params.append((ampspace[i], wlspace[i], 0.0))
            
        params = np.array(params)
        np.savetxt("results/params.csv", params, delimiter=",")
            
        for p in tqdm (range(int(train_count)), desc="Generating param responses with noise"):
            s = generate_microphone_pressure(params[p])
            s += np.random.normal(loc=0.0, scale=0.05, size=(34,))
            s = np.abs(s)
            scatters.append(s[indices[0]:indices[-1]])
            #plt.plot(s, label=str(params[p][1]))
        #plt.legend()
        #plt.show()

        params = np.array(params)
        scatters = np.array(scatters)

        plt.hist(params[:, 0], bins=50, alpha=0.5, label="amp")
        plt.legend()
        plt.show()
        np.savetxt("results/scatters.csv", scatters, delimiter=",")

    params = np.loadtxt("results/params.csv", delimiter=",")
    scatters = np.loadtxt("results/scatters.csv", delimiter=",")
    training_data = np.column_stack((params[:,:1], scatters))
    print("Combined params and scatter responses: ", training_data.shape)

    # Setup training data
    # Create pandas dataframe (currently hard coded to 2 param recovery! TODO: maybe make more generic)
    # Split into test and training sets
    columns = ['amp']
    for i in range(0, scatters.shape[1]):
        if i in indices:
            columns.append('r' + str(i))
    dataframe = pd.DataFrame(training_data,
                    columns=columns)

    # X3 contains the receiver data
    # Y3 contains the parameter data
    X3_train, X3_test, Y3_train, Y3_test = train_test_split(
        dataframe[columns[1*cosine_count:]], dataframe[columns[:1*cosine_count]], test_size=0.25, random_state=11)

    print("Training params view ", Y3_train.shape, ":\n", Y3_train, "\n")
    print("Training scatter view ", X3_train.shape, ":\n", X3_train, "\n")
    print("Testing params view:", Y3_test.shape, "\n", Y3_test, "\n")
    print("Testing scatter view:", X3_test.shape, "\n", X3_test, "\n")

    floatX = pytensor.config.floatX

    n_hidden_1 = 68
    n_hidden_2 = 68
    n_hidden_3 = 68

    #Initialize weightings between layers
    init_1 = np.random.randn(X3_train.shape[1], n_hidden_1).astype(floatX)
    init_2 = np.random.randn(n_hidden_1, n_hidden_2).astype(floatX)
    init_3 = np.random.randn(n_hidden_2, n_hidden_3).astype(floatX)
    #init_out = np.random.randn(n_hidden_2,Y3_train.shape[1]).astype(floatX)
    init_out = np.random.randn(n_hidden_2,1).astype(floatX)

    # Initialize random biases in each layer
    init_b_1 = np.random.randn(n_hidden_1).astype(floatX)
    init_b_2 = np.random.randn(n_hidden_2).astype(floatX)
    init_b_3 = np.random.randn(n_hidden_3).astype(floatX)
    init_b_out = np.random.randn(1).astype(floatX)

    model3_input = pytensor.shared(np.array(X3_train))
    model3_amp_output = pytensor.shared(np.array(Y3_train['amp']))
    #model3_wl_output = pytensor.shared(np.array(Y3_train['wl']))
    model3_output = pytensor.shared(np.array(Y3_train))

    # Define the Bayesian Neural Network
    class BayesianNN:
        def __init__(self, n_inputs, n_hidden):
            self.n_inputs = n_inputs
            self.n_hidden = n_hidden
            self.current_x = None
            self.current_y = None

        def build_model(self):
            with pm.Model() as self.model:
                # Prior distributions for weights and biases
                W1 = pm.Normal("W1", mu=0, sigma=1, shape=(self.n_inputs, self.n_hidden))
                b1 = pm.Normal("b1", mu=0, sigma=1, shape=(self.n_hidden,))
                
                W2 = pm.Normal("W2", mu=0, sigma=1, shape=(self.n_hidden, self.n_hidden))
                b2 = pm.Normal("b2", mu=0, sigma=1, shape=(self.n_hidden,))
                
                W3 = pm.Normal("W3", mu=0, sigma=1, shape=(self.n_hidden, self.n_hidden))
                b3 = pm.Normal("b3", mu=0, sigma=1, shape=(self.n_hidden,))
                
                W4 = pm.Normal("W4", mu=0, sigma=1, shape=(self.n_hidden, 1))
                b4 = pm.Normal("b4", mu=0, sigma=1, shape=(1,))
                
                # Input scatter data
                X = pm.Data("X", self.current_x.to_numpy())
                
                # Forward pass
                hidden1 = pm.math.tanh(pt.dot(X, W1) + b1)
                hidden2 = pm.math.tanh(pt.dot(hidden1, W2) + b2)
                hidden3 = pm.math.tanh(pt.dot(hidden2, W3) + b3)
                output = pt.dot(hidden3, W4) + b4
                
                # Likelihood from param data
                y = pm.Data("y", self.current_y.to_numpy())
                sigma = pm.HalfNormal("sigma", sigma=1)
                pm.Normal("amp", mu=output, sigma=sigma, observed=y)
        
        def train(self, train_x, train_y, draws=5000, tune=2000):
            self.current_x = train_x
            self.current_y = train_y
            self.build_model()
            with self.model:
                self.trace = pm.sample(draws=draws, tune=tune, chains=1, return_inferencedata=True)
                self.trace.to_netcdf("results/nn_trace.nc")
        
        def predict(self, X_new, Y_new):
            with self.model:
                pm.set_data({"X": X_new})
                pm.set_data({"y": Y_new})
                posterior_predictive = pm.sample_posterior_predictive(self.trace)["posterior_predictive"]
            return posterior_predictive["amp"]

    # Initialize and train model
    bnn = BayesianNN(n_inputs=X3_train.shape[1], n_hidden=68)
    bnn.train(X3_train, Y3_train)
    amps = np.array(bnn.predict(X3_test, Y3_test))

    amps_index = np.array(amps).squeeze()
    pred_amp = amps_index.mean(axis=0)

    plt.figure(figsize=(16, 9))
    plt.scatter(np.array(Y3_test["amp"]), np.array(pred_amp), label="Predicted vs test")
    plt.scatter(np.array(Y3_test["amp"]), np.array(Y3_test["amp"]), label="True test set")
    plt.legend()
    plt.show()

    from sklearn.metrics import r2_score
    print("R2 score: ", r2_score(np.array(Y3_test["amp"]), pred_amp))

    with pm.Model() as layer3_nn:
        tau = pm.HalfCauchy('tau', beta=1)

        # Weights from input to hidden layer
        s_i_1 = pm.HalfNormal('s_i_1', sigma=tau)
        weights_in_1 = pm.Normal('w_in_1', mu=0, sigma=s_i_1, shape=(model3_input.shape[1], n_hidden_1), initval=init_1)
        # Bias at 1st hidden layer
        bias_1 = pm.Normal('b_1', mu=0, sigma=1, shape=(n_hidden_1), initval=init_b_1)

        # Weights from 1st to 2nd hidden layer
        s_1_2 = pm.HalfNormal('s_1_2', sigma=tau)
        weights_1_2 = pm.Normal('w_1_2', mu=0, sigma=s_1_2, shape=(n_hidden_1, n_hidden_2), initval=init_2)
        # Bias at 2nd hidden layer
        bias_2 = pm.Normal('b_2', mu=0, sigma=1, shape=(n_hidden_2), initval=init_b_2)

        # Weights from 2nd to 3rd hidden layer
        s_2_3 = pm.HalfNormal('s_2_3', sigma=tau)
        weights_2_3 = pm.Normal('w_2_3', mu=0, sigma=s_2_3, shape=(n_hidden_2, n_hidden_3), initval=init_3)
        # Bias at 3rd hidden layer
        bias_3 = pm.Normal('b_3', mu=0, sigma=1, shape=(n_hidden_3), initval=init_b_3)

        # Weights from hidden layer to output (2 outputs) !!CHECK 1 OR 2 OUTPUT!!
        s_3_o = pm.HalfNormal('s_3_o', sigma=tau)
        weights_3_out = pm.Normal('w_3_out', mu=0, sigma=s_3_o, shape=(n_hidden_3,1), initval=init_out)
        # Bias at output hidden layer
        bias_out = pm.Normal('b_out', mu=0, sigma=1, shape=(1), initval=init_b_out)


        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(model3_input, weights_in_1) + bias_1)
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2) + bias_2)
        act_3 = pm.math.tanh(pm.math.dot(act_2, weights_2_3) + bias_3)
        act_out = pm.math.dot(act_3, weights_3_out) + bias_out
        #act_out_amp = pm.math.dot(act_3, weights_3_out[:,0]) + bias_out[0]
        #act_out_wl = pm.math.dot(act_3, weights_3_out[:,1]) + bias_out[1]

        #sigma_amp = pm.HalfNormal('sigma_amp', sigma=1)
        #out_amp = pm.Normal('amp', mu=act_out_amp, sigma=sigma_amp, observed=model3_amp_output)

        #sigma_wl = pm.HalfNormal('sigma_wl', sigma=1)
        #out_wl = pm.Normal('wl', mu=act_out_wl, sigma=sigma_wl, observed=model3_wl_output)

        #ONE PARAMETER OUTPUT
        sigma_amp = pm.HalfNormal('sigma_amp', sigma=1)
        out_amp = pm.TruncatedNormal('amp', lower=0.0, mu=act_out[0], sigma=sigma_amp, observed=model3_amp_output)

    # Linear regression first
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X3_train, Y3_train['amp'])
    print("Linear score: ", lr.score(X3_test, Y3_test['amp']))
    plt.scatter(Y3_train['amp'], Y3_train['amp'], label="True values")
    plt.scatter(Y3_train['amp'], lr.predict(X3_train), color="red", label="Model predictions")
    plt.legend()
    plt.show()

    run_model = True
    if run_model:
        with layer3_nn:

            s = pytensor.shared(pm.floatX(1.1))
            inference = pm.ADVI(cost_part_grad_scale=s)
            tracker = pm.callbacks.Tracker(
                mean=inference.approx.mean.eval,  # callable that returns mean
                std=inference.approx.std.eval,    # callable that returns std
            )

            approx = pm.fit(30_000, method=inference, callbacks=[tracker])
            nn_trace = approx.sample(10_000).sel(draw=slice(3000, None))

            #approx = inference.fit(n=100_000, callbacks=[tracker])
            #step = pm.NUTS(target_accept=0.9)
            #nn_trace = pm.sample(tune=2_000, draws=5_000, step=step, chains=1, cores=1, nuts_sampler="numpyro")
            nn_trace.to_netcdf("results/nn_trace.nc")

            fig = plt.figure(figsize=(16, 9))
            mu_ax = fig.add_subplot(221)
            std_ax = fig.add_subplot(222)
            hist_ax = fig.add_subplot(212)
            mu_ax.plot(tracker["mean"])
            mu_ax.set_title("Mean track")
            std_ax.plot(tracker["std"])
            std_ax.set_title("Std track")
            hist_ax.plot(inference.hist)
            hist_ax.set_title("Negative ELBO track")
            plt.savefig("results/nn_elbo.png")

            summary = az.summary(nn_trace)
            print(summary)
            with open("results/output.txt", "w") as txt:
                txt.write(summary.to_string())

    nn_trace = az.from_netcdf("results/nn_trace.nc")
    #print("Plotting forest")
    #az.plot_forest(nn_trace)
    #plt.show()

    # Replace shared variables with testing set
    # Need to set as shapes change when test and training sets differ in shape
    model3_input.set_value(np.array(X3_test).astype(floatX))
    model3_amp_output.set_value(np.array(Y3_test["amp"]).astype(floatX))
    #model3_wl_output.set_value(np.array(Y3_test["wl"]).astype(floatX))
    ppc_nn = pm.sample_posterior_predictive(nn_trace, var_names=["amp"], model=layer3_nn)["posterior_predictive"]

    from src.SymbolicMath import SymAngularMean
    amps = np.array(ppc_nn['amp'])
    amps_index = np.array(amps).squeeze()
    pred_amp = amps_index.mean(axis=0)

    plt.figure(figsize=(16, 9))
    plt.scatter(np.array(Y3_test["amp"]), np.array(pred_amp), label="Predicted vs test")
    plt.scatter(np.array(Y3_test["amp"]), np.array(Y3_test["amp"]), label="True test set")
    plt.legend()
    plt.show()
